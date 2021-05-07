"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference of Unanswerable VQA for submission and validation
"""
import argparse
import json
import os
from os.path import exists, join
from time import time

import torch
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd
import numpy as np
from cytoolz import concat

from data import (TokenBucketSampler, PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, UnansVqaEvalDataset, unans_vqa_eval_collate)
from model.unans import UniterForUnansVisualQuestionAnswering

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct, parse_with_config
from utils.const import BUCKET_SIZE, IMG_DIM


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    hps_file = join(f'{opts.output_dir}', 'log/hps.json')
    model_opts = Struct(json.load(open(hps_file)))

    # Prepare model
    if isinstance(opts.checkpoint, str) and exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = join(f'{opts.output_dir}', f'ckpt/model_step_{opts.checkpoint}.pt')
    LOGGER.info(f"Loading '{ckpt_file}'")
    checkpoint = torch.load(ckpt_file)
    model = UniterForUnansVisualQuestionAnswering.from_pretrained(
        join(f'{opts.output_dir}', 'log/model.json'), checkpoint,
        img_dim=IMG_DIM,
        unans_weight=model_opts.unans_weight,
        ans_threshold=model_opts.ans_threshold)
    model.to(device)
    if opts.fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')

    # load DBs and image dirs
    for name, txt_db, img_db in zip(opts.db_names, opts.txt_dbs, opts.img_dbs):
        eval_img_db = DetectFeatLmdb(img_db,
                                     model_opts.conf_th, model_opts.max_bb,
                                     model_opts.min_bb, model_opts.num_bb,
                                     opts.compressed_db)
        eval_txt_db = TxtTokLmdb(txt_db, -1)
        eval_dataset = UnansVqaEvalDataset(eval_txt_db, eval_img_db)

        sampler = TokenBucketSampler(eval_dataset.lens, bucket_size=BUCKET_SIZE,
                                     batch_size=opts.batch_size, droplast=False)
        eval_dataloader = DataLoader(eval_dataset,
                                     batch_sampler=sampler,
                                     num_workers=opts.n_workers,
                                     pin_memory=opts.pin_mem,
                                     collate_fn=unans_vqa_eval_collate)
        eval_dataloader = PrefetchLoader(eval_dataloader)

        val_log, results, logits = evaluate(model, eval_dataloader, name,
                                            opts.save_logits)
        result_dir = join(f'{opts.output_dir}', f'results_test/{name}')
        if not exists(result_dir) and rank == 0:
            os.makedirs(result_dir)

        all_results = list(concat(all_gather_list(results)))
        if opts.save_logits:
            all_logits = {}
            for id2logit in all_gather_list(logits):
                all_logits.update(id2logit)
        if hvd.rank() == 0:
            with open(join(f'{result_dir}',
                      f'stats_{opts.checkpoint}_all.json'), 'w') as f:
                json.dump(val_log, f)
            with open(join(f'{result_dir}',
                      f'results_{opts.checkpoint}_all.json'), 'w') as f:
                json.dump(all_results, f, indent=4, sort_keys=True)
            if opts.save_logits:
                np.savez(join(f'{result_dir}', f'logits_{opts.checkpoint}_all.npz'),
                         **all_logits)


@torch.no_grad()
def evaluate(model, eval_loader, name, save_logits=False):
    LOGGER.info(f"******* start running evaluation on {name} *******")
    model.eval()
    has_targets = False
    val_loss, tot_scores = 0.0, 0
    good_ones, good_zeros, pred_ones, gold_ones, gold_zeros = 0, 0, 0, 0, 0
    n_ex = 0
    st = time()
    results = []
    logits = {}
    for i, batch in enumerate(eval_loader):
        qids = batch['qids']
        loss, output = model(batch, classify=not save_logits)
        answers = output
        if save_logits:
            answers = model.classify_logits(output)
            output = output.squeeze(dim=-1).detach().cpu()
            for i, qid in enumerate(qids):
                logits[qid] = output[i].half().numpy()
        if 'targets' in batch:
            has_targets = True
            targets = batch['targets']
            correct = answers == targets
            val_loss += loss.sum().item()
            tot_scores += correct.sum().item()
            good_ones += (correct & (targets > 0)).sum().item()
            good_zeros += (correct & ~(targets > 0)).sum().item()
            pred_ones += answers.sum().item()
            gold_ones += targets.sum().item()
            gold_zeros += (~(targets > 0)).sum().item()
        for qid, answer in zip(qids, answers.squeeze(dim=-1).detach().cpu()):
            results.append({'answer': answer.item(), 'question_id': qid})
        if i % 100 == 0 and hvd.rank() == 0:
            n_results = len(results)
            n_results *= hvd.size()   # an approximation to avoid hangs
            LOGGER.info(f'  {n_results}/{len(eval_loader.dataset)} '
                        'answers predicted')
        n_ex += len(qids)
    n_ex = sum(all_gather_list(n_ex))
    if has_targets:
        val_loss = sum(all_gather_list(val_loss))
        tot_scores = sum(all_gather_list(tot_scores))
        good_ones = sum(all_gather_list(good_ones))
        good_zeros = sum(all_gather_list(good_zeros))
        pred_ones = sum(all_gather_list(pred_ones))
        gold_ones = sum(all_gather_list(gold_ones))
        gold_zeros = sum(all_gather_list(gold_zeros))
    tot_time = time()-st
    val_log = {}
    if has_targets:
        val_loss /= n_ex
        val_acc = tot_scores / n_ex
        val_precis = good_ones / pred_ones
        val_recall = good_ones / gold_ones
        val_f1 = (2 * val_precis * val_recall) / (val_precis + val_recall)
        val_log[f'loss'] = val_loss
        val_log[f'accuracy'] = round(val_acc * 100, 4)
        val_log[f'ans_accuracy'] = round(good_ones * 100 / gold_ones, 4)
        val_log[f'unans_accuracy'] = round(good_zeros * 100 / gold_zeros, 4)
        val_log[f'precision'] = round(val_precis, 6)
        val_log[f'recall'] = round(val_recall, 6)
        val_log[f'f1'] = round(val_f1, 6)
    model.train()
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds "
                f"at {int(n_ex/tot_time)} examples per second")
    if has_targets:
        LOGGER.info(f"  loss: {val_loss}")
        LOGGER.info(f"  accuracy: {val_acc*100:.4f}%, {tot_scores}/{n_ex}")
        LOGGER.info(f"  ans accuracy: {good_ones*100/gold_ones:.4f}%, {good_ones}/{gold_ones}")
        LOGGER.info(f"  unans accuracy: {good_zeros*100/gold_zeros:.4f}%, {good_zeros}/{gold_zeros}")
        LOGGER.info(f"  precision: {val_precis:.6f}, {good_ones}/{pred_ones}")
        LOGGER.info(f"  recall: {val_recall:.6f}, {good_ones}/{gold_ones}")
        LOGGER.info(f"  f1: {val_f1:.6f}")
    return val_log, results, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--db_names",
                        default=None, type=list,
                        help="The input datasets names.")
    parser.add_argument("--txt_dbs",
                        default=None, type=list,
                        help="The input train corpuses. (LMDB)")
    parser.add_argument("--img_dbs",
                        default=None, type=list,
                        help="The input train images.")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="can be the path to binary or int number (step)")
    parser.add_argument("--batch_size",
                        default=8192, type=int,
                        help="number of tokens in a batch")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory of the training command")
    parser.add_argument("--save_logits", action='store_true',
                        help="Whether to save logits (for making ensemble)")

    # device parameters
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    main(args)
