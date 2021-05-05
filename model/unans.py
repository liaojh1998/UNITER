"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VQA model
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel


class UniterForUnansVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for Unanswerable VQA
    """
    def __init__(self, config, img_dim, unans_weight=1.0, ans_threshold=0.5):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.unans_weight = unans_weight
        self.ans_threshold = ans_threshold
        self.unans_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 1)
        )
        self.apply(self.init_weights)

    def forward(self, batch, classify=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        answer_scores = self.unans_output(pooled_output)

        unans_loss = -1.0
        if 'targets' in batch:
            targets = batch['targets']
            weights = torch.ones(targets.shape, dtype=answer_scores.dtype)
            weights[targets == 0] = self.unans_weight
            unans_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets.type(answer_scores.dtype),
                weight=weights.to(targets.device), reduction='none')

        if classify:
            answers = torch.sigmoid(answer_scores) >= self.ans_threshold
            answers = answers.long()
            return unans_loss, answers

        return unans_loss
