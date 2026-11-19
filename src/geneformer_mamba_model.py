#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput


class GeneformerMambaConfig(PretrainedConfig):
    model_type = "geneformer_mamba"

    def __init__(
        self,
        vocab_size: int = 0,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        max_position_embeddings: int = 2048,
        pad_token_id: int = 0,
        hidden_dropout_prob: float = 0.02,
        num_labels: int = 2,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        pcw_auxiliary_loss_weight: float = 0.1,
        pcw_rank_loss_weight: float = 0.02,
        pcw_log2_min: float = 0.0,
        pcw_log2_max: float = 12.0,
        pcw_plateau_log2_init: float = 8.0,
        pcw_plateau_slope_init: float = 1.0,
        pcw_huber_beta: float = 0.5,
        use_expression_values: bool = True,
        num_cluster_labels: int = 0,
        num_lineage_labels: int = 0,
        num_study_labels: int = 0,
        num_orig_ident_labels: int = 0,
        num_multiome_labels: int = 0,
        cluster_auxiliary_loss_weight: float = 0.05,
        lineage_auxiliary_loss_weight: float = 0.02,
        study_adversarial_loss_weight: float = 0.02,
        orig_ident_adversarial_loss_weight: float = 0.02,
        multiome_adversarial_loss_weight: float = 0.02,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_labels = num_labels
        self.pcw_auxiliary_loss_weight = pcw_auxiliary_loss_weight
        self.pcw_rank_loss_weight = pcw_rank_loss_weight
        self.pcw_log2_min = pcw_log2_min
        self.pcw_log2_max = pcw_log2_max
        self.pcw_plateau_log2_init = pcw_plateau_log2_init
        self.pcw_plateau_slope_init = pcw_plateau_slope_init
        self.pcw_huber_beta = pcw_huber_beta
        self.use_expression_values = use_expression_values
        self.num_cluster_labels = num_cluster_labels
        self.num_lineage_labels = num_lineage_labels
        self.num_study_labels = num_study_labels
        self.num_orig_ident_labels = num_orig_ident_labels
        self.num_multiome_labels = num_multiome_labels
        self.cluster_auxiliary_loss_weight = cluster_auxiliary_loss_weight
        self.lineage_auxiliary_loss_weight = lineage_auxiliary_loss_weight
        self.study_adversarial_loss_weight = study_adversarial_loss_weight
        self.orig_ident_adversarial_loss_weight = orig_ident_adversarial_loss_weight
        self.multiome_adversarial_loss_weight = multiome_adversarial_loss_weight


def _load_mamba_cls():
    try:
        from mamba_ssm import Mamba
    except Exception as exc:
        raise ImportError(
            "mamba_ssm is required for GeneformerMamba. Install/activate it before running this task."
        ) from exc
    return Mamba


class GeneformerMambaBackbone(nn.Module):
    def __init__(self, config: GeneformerMambaConfig):
        super().__init__()
        Mamba = _load_mamba_cls()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.value_projection = nn.Linear(1, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(config.hidden_size),
                        "mamba": Mamba(
                            d_model=config.hidden_size,
                            d_state=config.state_size,
                            d_conv=config.conv_kernel,
                            expand=config.expand,
                        ),
                    }
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expression_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            input_ids = input_ids[:, : self.config.max_position_embeddings]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.config.max_position_embeddings]
            if expression_values is not None:
                expression_values = expression_values[:, : self.config.max_position_embeddings]
            seq_len = input_ids.shape[1]

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        if bool(self.config.use_expression_values) and expression_values is not None:
            hidden_states = hidden_states + self.value_projection(expression_values.to(hidden_states.dtype).unsqueeze(-1))
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = hidden_states + layer["mamba"](layer["norm"](hidden_states))
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)

        return self.final_norm(hidden_states)


class GeneformerMambaPreTrainedModel(PreTrainedModel):
    config_class = GeneformerMambaConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GeneformerMambaForMaskedLM(GeneformerMambaPreTrainedModel):
    def __init__(self, config: GeneformerMambaConfig):
        super().__init__(config)
        self.backbone = GeneformerMambaBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


def _pool_hidden_states(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _pool_hvg_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    auxiliary_gene_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if auxiliary_gene_mask is None:
        return _pool_hidden_states(hidden_states, attention_mask)
    mask = auxiliary_gene_mask
    if attention_mask is not None:
        mask = mask * attention_mask.to(mask.dtype)
    mask = mask.unsqueeze(-1).to(hidden_states.dtype)
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    fallback = _pool_hidden_states(hidden_states, attention_mask)
    has_hvg = mask.sum(dim=1) > 0
    return torch.where(has_hvg, pooled, fallback)


def _pairwise_rank_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.numel() < 2:
        return predicted.new_tensor(0.0)
    target_diff = target[:, None] - target[None, :]
    pred_diff = predicted[:, None] - predicted[None, :]
    valid = target_diff.abs() > 1e-6
    if not bool(valid.any()):
        return predicted.new_tensor(0.0)
    direction = target_diff.sign()
    return F.softplus(-direction[valid] * pred_diff[valid]).mean()


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def gradient_reverse(input_tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(input_tensor, scale)


class LearnablePCWPlateauScale(nn.Module):
    """Monotone log2(PCW)-based scale that can learn a late plateau."""

    def __init__(self, config: GeneformerMambaConfig):
        super().__init__()
        log2_min = float(config.pcw_log2_min)
        log2_max = max(float(config.pcw_log2_max), log2_min + 1e-3)
        init_mid = min(max(float(config.pcw_plateau_log2_init), log2_min), log2_max)
        init_slope = max(float(config.pcw_plateau_slope_init), 1e-3)
        init_fraction = (init_mid - log2_min) / (log2_max - log2_min)
        init_fraction = min(max(init_fraction, 1e-4), 1.0 - 1e-4)
        self.register_buffer("log2_min", torch.tensor(log2_min, dtype=torch.float32))
        self.register_buffer("log2_range", torch.tensor(log2_max - log2_min, dtype=torch.float32))
        self.midpoint_logit = nn.Parameter(torch.logit(torch.tensor(init_fraction)))
        self.raw_slope = nn.Parameter(torch.log(torch.expm1(torch.tensor(init_slope))))
        self.raw_amplitude = nn.Parameter(torch.log(torch.expm1(torch.tensor(2.0))))
        self.offset = nn.Parameter(torch.tensor(-1.0))

    def forward(self, pcw_values: torch.Tensor) -> torch.Tensor:
        log2_pcw = torch.log2(pcw_values.to(dtype=torch.float32).clamp(min=torch.finfo(torch.float32).tiny))
        midpoint = self.log2_min + torch.sigmoid(self.midpoint_logit) * self.log2_range
        slope = F.softplus(self.raw_slope) + 1e-4
        amplitude = F.softplus(self.raw_amplitude) + 1e-4
        return self.offset + amplitude * torch.sigmoid(slope * (log2_pcw - midpoint))

    def parameters_as_dict(self) -> dict[str, float]:
        midpoint = self.log2_min + torch.sigmoid(self.midpoint_logit) * self.log2_range
        return {
            "equation": "target = offset + amplitude * sigmoid(slope * (log2(PCW) - midpoint_log2_pcw))",
            "offset": float(self.offset.detach().cpu()),
            "amplitude": float((F.softplus(self.raw_amplitude) + 1e-4).detach().cpu()),
            "slope": float((F.softplus(self.raw_slope) + 1e-4).detach().cpu()),
            "midpoint_log2_pcw": float(midpoint.detach().cpu()),
            "midpoint_pcw": float(torch.pow(torch.tensor(2.0), midpoint).detach().cpu()),
        }


class GeneformerMambaForMaskedLMWithPCW(GeneformerMambaPreTrainedModel):
    def __init__(self, config: GeneformerMambaConfig):
        super().__init__(config)
        self.backbone = GeneformerMambaBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pcw_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )
        self.pcw_scale = LearnablePCWPlateauScale(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expression_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pcw_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expression_values=expression_values,
        )
        logits = self.lm_head(hidden_states)
        total_loss = None
        if labels is not None:
            total_loss = CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if pcw_values is not None:
            pooled = _pool_hidden_states(hidden_states, attention_mask)
            predicted = self.pcw_head(pooled).squeeze(-1)
            target = self.pcw_scale(pcw_values).to(predicted.device)
            pcw_loss = SmoothL1Loss(beta=float(self.config.pcw_huber_beta))(predicted, target)
            rank_loss = _pairwise_rank_loss(predicted, target)
            auxiliary_loss = (
                float(self.config.pcw_auxiliary_loss_weight) * pcw_loss
                + float(self.config.pcw_rank_loss_weight) * rank_loss
            )
            total_loss = auxiliary_loss if total_loss is None else total_loss + auxiliary_loss

        return MaskedLMOutput(loss=total_loss, logits=logits, hidden_states=None, attentions=None)


class GeneformerMambaForSequenceClassification(GeneformerMambaPreTrainedModel):
    def __init__(self, config: GeneformerMambaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.backbone = GeneformerMambaBackbone(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        expression_values = kwargs.get("expression_values")
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expression_values=expression_values,
        )
        pooled = _pool_hidden_states(hidden_states, attention_mask)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            if self.config.problem_type == "regression" or self.num_labels == 1:
                loss = MSELoss()(logits.squeeze(), labels.squeeze().to(logits.dtype))
            else:
                loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


class GeneformerMambaForAtlasPretraining(GeneformerMambaPreTrainedModel):
    def __init__(self, config: GeneformerMambaConfig):
        super().__init__(config)
        self.backbone = GeneformerMambaBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.cluster_head = (
            nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Dropout(config.hidden_dropout_prob), nn.Linear(config.hidden_size, config.num_cluster_labels))
            if config.num_cluster_labels > 0 else None
        )
        self.lineage_head = (
            nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Dropout(config.hidden_dropout_prob), nn.Linear(config.hidden_size, config.num_lineage_labels))
            if config.num_lineage_labels > 0 else None
        )
        self.study_head = (
            nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Dropout(config.hidden_dropout_prob), nn.Linear(config.hidden_size, config.num_study_labels))
            if config.num_study_labels > 0 else None
        )
        self.orig_ident_head = (
            nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Dropout(config.hidden_dropout_prob), nn.Linear(config.hidden_size, config.num_orig_ident_labels))
            if config.num_orig_ident_labels > 0 else None
        )
        self.multiome_head = (
            nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Dropout(config.hidden_dropout_prob), nn.Linear(config.hidden_size, config.num_multiome_labels))
            if config.num_multiome_labels > 0 else None
        )
        self.pcw_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )
        self.pcw_scale = LearnablePCWPlateauScale(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expression_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        cluster_labels: Optional[torch.Tensor] = None,
        lineage_labels: Optional[torch.Tensor] = None,
        study_labels: Optional[torch.Tensor] = None,
        orig_ident_labels: Optional[torch.Tensor] = None,
        multiome_labels: Optional[torch.Tensor] = None,
        pcw_values: Optional[torch.Tensor] = None,
        auxiliary_gene_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expression_values=expression_values,
        )
        pooled = _pool_hidden_states(hidden_states, attention_mask)
        auxiliary_pooled = _pool_hvg_hidden_states(hidden_states, attention_mask, auxiliary_gene_mask)
        logits = self.lm_head(hidden_states)

        total_loss = None
        if labels is not None:
            total_loss = CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if self.cluster_head is not None and cluster_labels is not None:
            cluster_loss = CrossEntropyLoss(ignore_index=-1)(self.cluster_head(auxiliary_pooled), cluster_labels.view(-1))
            total_loss = cluster_loss * float(self.config.cluster_auxiliary_loss_weight) if total_loss is None else total_loss + cluster_loss * float(self.config.cluster_auxiliary_loss_weight)

        if self.lineage_head is not None and lineage_labels is not None:
            lineage_loss = CrossEntropyLoss(ignore_index=-1)(self.lineage_head(auxiliary_pooled), lineage_labels.view(-1))
            total_loss = lineage_loss * float(self.config.lineage_auxiliary_loss_weight) if total_loss is None else total_loss + lineage_loss * float(self.config.lineage_auxiliary_loss_weight)

        if self.study_head is not None and study_labels is not None:
            reversed_pooled = gradient_reverse(pooled, 1.0)
            study_loss = CrossEntropyLoss(ignore_index=-1)(self.study_head(reversed_pooled), study_labels.view(-1))
            total_loss = study_loss * float(self.config.study_adversarial_loss_weight) if total_loss is None else total_loss + study_loss * float(self.config.study_adversarial_loss_weight)

        if self.orig_ident_head is not None and orig_ident_labels is not None:
            reversed_pooled = gradient_reverse(pooled, 1.0)
            orig_ident_loss = CrossEntropyLoss(ignore_index=-1)(self.orig_ident_head(reversed_pooled), orig_ident_labels.view(-1))
            total_loss = orig_ident_loss * float(self.config.orig_ident_adversarial_loss_weight) if total_loss is None else total_loss + orig_ident_loss * float(self.config.orig_ident_adversarial_loss_weight)

        if self.multiome_head is not None and multiome_labels is not None:
            reversed_pooled = gradient_reverse(pooled, 1.0)
            multiome_loss = CrossEntropyLoss(ignore_index=-1)(self.multiome_head(reversed_pooled), multiome_labels.view(-1))
            total_loss = multiome_loss * float(self.config.multiome_adversarial_loss_weight) if total_loss is None else total_loss + multiome_loss * float(self.config.multiome_adversarial_loss_weight)

        if pcw_values is not None and (
            float(self.config.pcw_auxiliary_loss_weight) > 0.0
            or float(self.config.pcw_rank_loss_weight) > 0.0
        ):
            finite = torch.isfinite(pcw_values) & (pcw_values > 0)
            if bool(finite.any()):
                predicted = self.pcw_head(pooled[finite]).squeeze(-1)
                target = self.pcw_scale(pcw_values[finite]).to(predicted.device)
                pcw_loss = SmoothL1Loss(beta=float(self.config.pcw_huber_beta))(predicted, target)
                rank_loss = _pairwise_rank_loss(predicted, target)
                auxiliary_loss = (
                    float(self.config.pcw_auxiliary_loss_weight) * pcw_loss
                    + float(self.config.pcw_rank_loss_weight) * rank_loss
                )
                total_loss = auxiliary_loss if total_loss is None else total_loss + auxiliary_loss

        return MaskedLMOutput(loss=total_loss, logits=logits, hidden_states=None, attentions=None)
