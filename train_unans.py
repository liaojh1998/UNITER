"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for Unanswerable VQA

This code is based on the code for VQA
"""
import argparse
import json
import os
from os.path import abspath, dirname, exists, join
from time import time

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam, Adamax

from apex import amp
from horovod import torch as hvd
from pytorch_pretrained_bert import BertTokenizer

from tqdm import tqdm

from data import (TokenBucketSampler, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens,
                  UnansVqaDataset, UnansVqaEvalDataset,
                  unans_vqa_collate, unans_vqa_eval_collate)
from model.unans import UniterForUnansVisualQuestionAnswering
from optim import AdamW, get_lr_sched

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import BUCKET_SIZE, IMG_DIM


def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=is_train)
    dataloader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def build_optimizer(model, opts):
    """ vqa linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'vqa_output' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'vqa_output' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    # load DBs and image dirs
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)
    # train
    LOGGER.info(f"Loading Train Dataset "
                f"{opts.train_txt_dbs}, {opts.train_img_dbs}")
    train_datasets = []
    for txt_path, img_path in zip(opts.train_txt_dbs, opts.train_img_dbs):
        img_db = all_img_dbs[img_path]
        txt_db = TxtTokLmdb(txt_path, opts.max_txt_len)
        dataset = UnansVqaDataset(txt_db, img_db)
        if args.train_subset < 1.0:
            num = round(args.train_subset * len(dataset))
            LOGGER.info(f"Using {num} of {len(dataset)} examples from '{txt_path}'")
            indices = torch.randperm(len(dataset))[:num].numpy()
            dataset.subset(indices)
        train_datasets.append(dataset)
    train_dataset = ConcatDatasetWithLens(train_datasets)
    LOGGER.info(f"Train dataset contains {len(train_dataset)} examples")
    train_dataloader = None
    if len(train_dataset) > 0:
        train_dataloader = build_dataloader(train_dataset, unans_vqa_collate, True, opts)
    else:
        LOGGER.info(f"There's no training data, running validation on checkpoint instead")

    # val
    LOGGER.info(f"Loading Validation Dataset {opts.val_txt_db}, {opts.val_img_db}")
    val_img_db = all_img_dbs[opts.val_img_db]
    val_txt_db = TxtTokLmdb(opts.val_txt_db, -1)
    val_dataset = UnansVqaEvalDataset(val_txt_db, val_img_db)
    val_dataloader = build_dataloader(val_dataset, unans_vqa_eval_collate,
                                      False, opts)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}

    all_dbs = opts.train_txt_dbs + [opts.val_txt_db]
    toker = json.load(open(f'{all_dbs[0]}/meta.json'))['toker']
    assert all(toker == json.load(open(f'{db}/meta.json'))['toker']
               for db in all_dbs)
    if opts.verbose:
        toker = BertTokenizer.from_pretrained(
            toker, do_lower_case='uncased' in toker)
    model = UniterForUnansVisualQuestionAnswering.from_pretrained(
        opts.model_config, checkpoint,
        img_dim=IMG_DIM,
        unans_weight=opts.unans_weight, ans_threshold=opts.ans_threshold)
    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')
    global_step = 0
    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        if len(train_dataset) > 0:
            pbar = tqdm(total=opts.num_train_steps)
        else:
            pbar = NoOp()
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        os.makedirs(join(opts.output_dir, 'results'))  # store VQA predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataset) * hvd.size())
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)
    LOGGER.info("***** Unanswerable VQA Configs *****")
    LOGGER.info("  unans_weight = %f", opts.unans_weight)
    LOGGER.info("  ans_threshold = %f", opts.ans_threshold)
    LOGGER.info("  train_subset = %f", opts.train_subset)

    running_loss = RunningMeter('loss')
    model.train()
    n_examples = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()

    # training
    TB_LOGGER.step()
    if train_dataloader is not None:
        step = 0
        while True:
            for batch in train_dataloader:
                n_examples += batch['input_ids'].size(0)

                if opts.verbose:
                    LOGGER.info(f"***** Step {step} *****")
                    LOGGER.info(f"  Batch input_ids shape = {batch['input_ids'].shape}")
                    LOGGER.info(f"  Batch img_feat shape = {batch['img_feat'].shape}")
                    LOGGER.info(f"  Batch img_pos_feat shape = {batch['img_pos_feat'].shape}")
                    LOGGER.info(f"  Batch targets shape = {batch['targets'].shape}")

                    ex_str = toker.convert_ids_to_tokens(batch['input_ids'][0].detach().cpu().numpy())
                    LOGGER.info(f"  Batch 1st example str = '{ex_str}'")
                    LOGGER.info(f"  Batch 1st example img = '{batch['img_feat'][0]}'")
                    LOGGER.info(f"  Batch 1st example img_pos = '{batch['img_pos_feat'][0]}'")

                loss, _ = model(batch, classify=False)
                loss = loss.mean() * batch['targets'].size(1)  # instance-leval bce
                delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
                with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                    ) as scaled_loss:
                    scaled_loss.backward()
                    if not delay_unscale:
                        # gather gradients from every processes
                        # do this before unscaling to make sure every process uses
                        # the same gradient scale
                        grads = [p.grad.data for p in model.parameters()
                                 if p.requires_grad and p.grad is not None]
                        all_reduce_and_rescale_tensors(grads, float(1))

                running_loss(loss.item())

                if (step + 1) % opts.gradient_accumulation_steps == 0:
                    global_step += 1

                    # learning rate scheduling
                    lr_this_step = get_lr_sched(global_step, opts)
                    for i, param_group in enumerate(optimizer.param_groups):
                        if i == 0 or i == 1:
                            param_group['lr'] = lr_this_step * opts.lr_mul
                        elif i == 2 or i == 3:
                            param_group['lr'] = lr_this_step
                        else:
                            raise ValueError()
                    TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                    # log loss
                    # NOTE: not gathered across GPUs for efficiency
                    TB_LOGGER.add_scalar('loss', running_loss.val, global_step)

                    # update model params
                    if opts.grad_norm != -1:
                        grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                    opts.grad_norm)
                        TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                    if global_step % 100 == 0:
                        # monitor training throughput
                        LOGGER.info(f'============Step {global_step}=============')
                        tot_ex = sum(all_gather_list(n_examples))
                        ex_per_sec = int(tot_ex / (time()-start))
                        LOGGER.info(f'{tot_ex} examples trained at '
                                    f'{ex_per_sec} ex/s')
                        TB_LOGGER.add_scalar('perf/ex_per_s',
                                             ex_per_sec, global_step)
                        LOGGER.info(f'===========================================')

                    if global_step % opts.save_steps == 0:
                        model_saver.save(model, global_step)

                    if global_step % opts.log_steps == 0:
                        val_log, results = validate(model, val_dataloader, "valid")
                        with open(f'{opts.output_dir}/results/'
                                  f'val_results_{global_step}_'
                                  f'rank{rank}.json', 'w') as f:
                            json.dump(results, f)
                        TB_LOGGER.log_scaler_dict(val_log)
                        train_log, results = validate(model, train_dataloader, "train")
                        with open(f'{opts.output_dir}/results/'
                                  f'train_results_{global_step}_'
                                  f'rank{rank}.json', 'w') as f:
                            json.dump(results, f)
                        TB_LOGGER.log_scaler_dict(train_log)

                    TB_LOGGER.step()

                step += 1
                if global_step >= opts.num_train_steps:
                    break

            if global_step >= opts.num_train_steps:
                break
            n_epoch += 1
            LOGGER.info(f"finished {n_epoch} epochs")

    # validation
    if train_dataloader is None or opts.num_train_steps % opts.save_steps != 0:
        model_saver.save(model, global_step)
        val_log, results = validate(model, val_dataloader, "valid")
        with open(f'{opts.output_dir}/results/'
                  f'val_results_{global_step}_'
                  f'rank{rank}.json', 'w') as f:
            json.dump(results, f)
        TB_LOGGER.log_scaler_dict(val_log)
        if train_dataloader is not None:
            train_log, results = validate(model, train_dataloader, "train")
            with open(f'{opts.output_dir}/results/'
                      f'train_results_{global_step}_'
                      f'rank{rank}.json', 'w') as f:
                json.dump(results, f)
            TB_LOGGER.log_scaler_dict(train_log)


@torch.no_grad()
def validate(model, val_loader, name="valid"):
    LOGGER.info(f"start running {name} set validation...")
    model.eval()
    val_loss = 0
    tot_score = 0
    good_ones = 0
    good_zeros = 0
    pred_ones = 0
    gold_ones = 0
    gold_zeros = 0
    n_ex = 0
    st = time()
    results = {}
    for i, batch in enumerate(val_loader):
        loss, answers = model(batch, classify=True)
        val_loss += loss.sum().item()
        targets = batch['targets']
        correct = answers == targets
        tot_score += correct.sum().item()
        good_ones += (correct & (targets > 0)).sum().item()
        good_zeros += (correct & ~(targets > 0)).sum().item()
        pred_ones += answers.sum().item()
        gold_ones += targets.sum().item()
        gold_zeros += (~(targets > 0)).sum().item()
        for qid, answer in zip(batch['qids'], answers.squeeze(dim=-1)):
            results[qid] = answer.item()
        n_ex += len(batch['qids'])
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    good_ones = sum(all_gather_list(good_ones))
    good_zeros = sum(all_gather_list(good_zeros))
    pred_ones = sum(all_gather_list(pred_ones))
    gold_ones = sum(all_gather_list(gold_ones))
    gold_zeros = sum(all_gather_list(gold_zeros))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_precis = good_ones / pred_ones
    val_recall = good_ones / gold_ones
    val_f1 = (2 * val_precis * val_recall) / (val_precis + val_recall)
    val_log = {f'{name}_eval/loss': val_loss,
               f'{name}_eval/accuracy': val_acc,
               f'{name}_eval/ans_accuracy': good_ones / gold_ones,
               f'{name}_eval/unans_accuracy': good_zeros / gold_zeros,
               f'{name}_eval/precision': val_precis,
               f'{name}_eval/recall': val_recall,
               f'{name}_eval/f1': val_f1,
               f'{name}_eval/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"validation finished in {int(tot_time)} seconds")
    LOGGER.info(f"accuracy: {val_acc*100:.2f}%, {tot_score}/{n_ex}")
    LOGGER.info(f"ans accuracy: {good_ones*100/gold_ones:.2f}%, {good_ones}/{gold_ones}")
    LOGGER.info(f"unans accuracy: {good_zeros*100/gold_zeros:.2f}%, {good_zeros}/{gold_zeros}")
    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_mul", default=10.0, type=float,
                        help="multiplier for top layer lr")
    parser.add_argument("--save_steps", default=1000, type=int,
                        help="Run save every X steps")
    parser.add_argument("--log_steps", default=500, type=int,
                        help="Run logging every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")

    # unans model parameters
    parser.add_argument("--unans_weight", default=1.0, type=float,
                        help="Weight applied to unanswerable questions "
                             "loss to promote class balance in learning")
    parser.add_argument("--ans_threshold", default=0.5, type=float,
                        help="Threshold for an answer prediction"
                             "probability to be considered as answerable")
    parser.add_argument("--train_subset", default=1.0, type=float,
                        help="Percent of the training data to use to "
                             "train the model")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose output")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
