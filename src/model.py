# -*- coding: utf-8 -*-
"""
model.py
========
Kien truc mo hinh PhoBERT-AMP (Adaptive Metadata-aware PhoBERT):

  Nhanh van ban:
    PhoBERT (Contextual Embedding) -> Multi-Kernel CNN (Conv1D + BatchNorm
    + ReLU) -> Global Max Pooling -> Vector van ban

  Nhanh metadata (Per-Field Embedding):
    Moi truong metadata (rating, num_likes, quality) -> Embedding rieng
    (Dense + ReLU) -> Concat tat ca -> Vector metadata

  Adaptive Metadata Gate:
    concat(text_vec, meta_vec) -> MLP 2 lop -> Sigmoid -> gate_score
    gated_metadata = gate_score * meta_vec
    (gate quyet dinh muc do su dung metadata cho tung sample)

  Fusion:
    Concatenate(text_vec, gated_metadata) -> LayerNorm -> FC + ReLU
    -> Dropout -> Linear(num_classes) [Softmax ap dung o ngoai]

Ghi chu ve Softmax: mo hinh tra ve LOGITS THO (chua qua softmax). Khi train,
`nn.CrossEntropyLoss` da tu dong ap dung log-softmax ben trong nen KHONG
duoc goi them softmax truoc khi tinh loss. Khi predict/inference, goi
`torch.softmax(logits, dim=-1)` de ra xac suat 3 lop.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MultiKernelCNNBranch(nn.Module):
    """Multi-Kernel CNN + BatchNorm + Global Max Pooling tren embedding PhoBERT.

    Dua tren TextCNN (Kim, 2014): nhieu bo loc (filter) voi kich thuoc khac
    nhau (2,3,4,5 tu lien tiep) de bat cac cum tu tieu cuc/tich cuc/mau
    ngon ngu o nhieu do dai khac nhau.

    Cai tien so voi TextCNN goc: them BatchNorm1d sau moi Conv1d (truoc ReLU)
    de on dinh gradient va tang toc hoi tu.
    """

    def __init__(self, embed_dim: int, filter_sizes: List[int], num_filters: int):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                    padding=0,
                )
                for fs in filter_sizes
            ]
        )
        # BatchNorm1d cho moi kernel size - chuan hoa output sau Conv1d
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_filters) for _ in filter_sizes]
        )
        self.filter_sizes = filter_sizes
        self.output_dim = num_filters * len(filter_sizes)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, seq_len, embed_dim) - output cua PhoBERT
            attention_mask: (batch, seq_len) - 1 cho token that, 0 cho padding
        Returns:
            (batch, num_filters * len(filter_sizes))
        """
        # Conv1d ky vong (batch, channels, seq_len) nen phai hoan vi truc
        x = embeddings.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        pooled_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            fs = conv.kernel_size[0]

            # Neu cau ngan hon kernel size, pad them de Conv1d khong loi.
            if x.size(2) < fs:
                pad_amount = fs - x.size(2)
                x_in = F.pad(x, (0, pad_amount))
                mask_1d = F.pad(attention_mask.float(), (0, pad_amount), value=0.0)
            else:
                x_in = x
                mask_1d = attention_mask.float()

            conv_out = conv(x_in)       # (batch, num_filters, L_out)
            conv_out = bn(conv_out)     # BatchNorm truoc ReLU
            conv_out = F.relu(conv_out)
            L_out = conv_out.size(2)

            # Tinh mask chinh xac cho output sau Conv1d:
            # Dung min-pool tren mask (1=real, 0=pad) voi kernel_size=fs
            mask_for_conv = mask_1d.unsqueeze(1)  # (batch, 1, seq_len)
            inv_mask = 1.0 - mask_for_conv
            if inv_mask.size(2) >= fs:
                inv_pooled = F.max_pool1d(inv_mask, kernel_size=fs, stride=1, padding=0)
            else:
                inv_pooled = inv_mask
            mask_out = 1.0 - inv_pooled  # (batch, 1, L_out_mask)

            # Dam bao mask_out va conv_out cung do dai
            if mask_out.size(2) > L_out:
                mask_out = mask_out[:, :, :L_out]
            elif mask_out.size(2) < L_out:
                mask_out = F.pad(mask_out, (0, L_out - mask_out.size(2)), value=0.0)

            # Che cac vi tri padding bang -inf TRUOC max-pool
            conv_out = conv_out.masked_fill(mask_out == 0, float("-inf"))

            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            # Phong ho truong hop toan bo vi tri bi mask
            pooled = torch.nan_to_num(pooled, neginf=0.0, posinf=0.0)
            pooled_outputs.append(pooled)  # (batch, num_filters)

        return torch.cat(pooled_outputs, dim=1)  # (batch, num_filters * n_sizes)


class MetadataFieldEmbedding(nn.Module):
    """Per-Field Metadata Embedding.

    Thay vi dung 1 Dense layer chung cho toan bo vector metadata (3-dim ->
    64-dim), moi truong metadata (rating, num_likes, quality) co RIENG 1
    embedding layer (1 -> per_field_dim). Sau do concat tat ca.

    Loi ich: model hoc bieu dien RIENG cho tung truong, khong bi "tron lan"
    ngay tu dau. Vi du, "rating cao" va "nhieu luot thich" duoc encode
    doc lap truoc khi ket hop.
    """

    def __init__(self, num_fields: int, per_field_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.field_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, per_field_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_fields)
            ]
        )
        self.num_fields = num_fields
        self.per_field_dim = per_field_dim
        self.output_dim = num_fields * per_field_dim

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metadata: (batch, num_fields) - vector metadata da chuan hoa
        Returns:
            (batch, num_fields * per_field_dim)
        """
        field_outputs = []
        for i, embed in enumerate(self.field_embeddings):
            field_input = metadata[:, i:i+1]  # (batch, 1)
            field_outputs.append(embed(field_input))  # (batch, per_field_dim)
        return torch.cat(field_outputs, dim=1)  # (batch, num_fields * per_field_dim)


class AdaptiveMetadataGate(nn.Module):
    """Adaptive Metadata Gate - co che hoc trong so metadata.

    Gate quyet dinh MUC DO su dung metadata cho TUNG SAMPLE (khong phai
    toan cuc). Dieu nay cho phep model:
      - Khi text DA RO RANG (vd "san pham rat te"): gate ~ 0, bo qua metadata
      - Khi text MO HO (vd "ok"): gate ~ 1, su dung metadata (rating, likes)
        de ho tro phan loai

    Cong thuc:
        gate_input = concat(text_vector, metadata_vector)
        gate_score = sigmoid(W2 * ReLU(W1 * gate_input + b1) + b2) * gate_max_val
        gated_metadata = gate_score * metadata_vector

    gate_score la scalar trong (0, gate_max_val) cho moi sample trong batch.
    """

    def __init__(self, text_dim: int, metadata_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.2, gate_max_val: float = 0.5):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(text_dim + metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.gate_max_val = gate_max_val

    def forward(self, text_vector: torch.Tensor, metadata_vector: torch.Tensor):
        """
        Args:
            text_vector: (batch, text_dim)
            metadata_vector: (batch, metadata_dim)
        Returns:
            gated_metadata: (batch, metadata_dim) - metadata da duoc gate
            gate_score: (batch, 1) - trong so gate (de log va regularize)
        """
        gate_input = torch.cat([text_vector, metadata_vector], dim=1)
        gate_score = self.gate_net(gate_input)  # (batch, 1)
        # Cap gate score bang cach scale voi gate_max_val
        gate_score = gate_score * self.gate_max_val
        gated_metadata = gate_score * metadata_vector  # (batch, metadata_dim)
        return gated_metadata, gate_score


class PhoBertAMP(nn.Module):
    """PhoBERT-AMP: Adaptive Metadata-aware PhoBERT.

    Ket hop 3 module chinh:
      1. Multi-Kernel CNN Branch: trich xuat dac trung van ban tu PhoBERT
      2. Per-Field Metadata Embedding: embed tung truong metadata rieng
      3. Adaptive Metadata Gate: hoc trong so de quyet dinh muc do dung metadata

    Args:
        phobert_model_name: ten model tren HuggingFace Hub
        num_classes: so nhan dau ra (3: negative/normal/positive)
        cnn_filter_sizes, cnn_num_filters: cau hinh nhanh Multi-Kernel CNN
        metadata_dim: so truong metadata dau vao (3: rating, num_likes, quality)
        per_field_embed_dim: so chieu embed cho MOI truong metadata
        gate_hidden_dim: so chieu hidden cua gate MLP
        gate_dropout: dropout trong gate MLP
        fc_hidden_dim: so chieu FC sau fusion
        dropout: dropout truoc classifier
        freeze_phobert: dong bang PhoBERT (chi train head)
    """

    def __init__(
        self,
        phobert_model_name: str = "vinai/phobert-base",
        num_classes: int = 3,
        cnn_filter_sizes: List[int] = (2, 3, 4, 5),
        cnn_num_filters: int = 128,
        metadata_dim: int = 3,
        per_field_embed_dim: int = 32,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.2,
        fc_hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_phobert: bool = False,
        gate_max_val: float = 0.5,
        metadata_dropout: float = 0.3,
    ):
        super().__init__()
        self.metadata_dropout = metadata_dropout

        # --- PhoBERT backbone ---
        self.phobert = AutoModel.from_pretrained(phobert_model_name)
        hidden_size = self.phobert.config.hidden_size

        if freeze_phobert:
            for param in self.phobert.parameters():
                param.requires_grad = False

        # --- Nhanh van ban: Multi-Kernel CNN ---
        self.text_branch = MultiKernelCNNBranch(
            embed_dim=hidden_size,
            filter_sizes=list(cnn_filter_sizes),
            num_filters=cnn_num_filters,
        )

        # --- Nhanh metadata: Per-Field Embedding ---
        self.metadata_branch = MetadataFieldEmbedding(
            num_fields=metadata_dim,
            per_field_dim=per_field_embed_dim,
            dropout=0.1,
        )

        # --- Adaptive Metadata Gate ---
        metadata_embed_dim = self.metadata_branch.output_dim
        self.metadata_gate = AdaptiveMetadataGate(
            text_dim=self.text_branch.output_dim,
            metadata_dim=metadata_embed_dim,
            hidden_dim=gate_hidden_dim,
            dropout=gate_dropout,
            gate_max_val=gate_max_val,
        )

        # --- Fusion head: LayerNorm + FC + Dropout + Classifier ---
        fusion_dim = self.text_branch.output_dim + metadata_embed_dim
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.fc = nn.Linear(fusion_dim, fc_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc_hidden_dim, num_classes)

        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            Khi training: (logits, gate_scores)
                - logits: (batch, num_classes) - CHUA qua softmax
                - gate_scores: (batch, 1) - trong so gate (de regularize)
            Khi eval: logits only (tuong thich nguoc voi pipeline cu)
        """
        # --- PhoBERT encoding ---
        phobert_out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = phobert_out.last_hidden_state  # (batch, seq_len, hidden)

        # --- Nhanh van ban ---
        text_vector = self.text_branch(embeddings, attention_mask)  # (batch, cnn_out)

        # --- Nhanh metadata ---
        metadata_vector = self.metadata_branch(metadata)  # (batch, meta_embed)

        # Apply metadata dropout during training to prevent shortcut learning
        if self.training and self.metadata_dropout > 0:
            mask = (torch.rand(metadata_vector.size(0), 1, device=metadata_vector.device) >= self.metadata_dropout).float()
            metadata_vector = metadata_vector * mask

        # --- Adaptive Metadata Gate ---
        gated_metadata, gate_scores = self.metadata_gate(
            text_vector, metadata_vector
        )  # (batch, meta_embed), (batch, 1)

        # --- Fusion ---
        fused = torch.cat([text_vector, gated_metadata], dim=1)
        fused = self.layer_norm(fused)
        hidden = F.relu(self.fc(fused))
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        if self.training:
            return logits, gate_scores
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Tien ich: forward + softmax, dung khi inference (khong tinh grad)."""
        self.eval()
        logits = self.forward(input_ids, attention_mask, metadata)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_with_gate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict + tra ve gate_score (dung cho API/debug).

        Returns:
            probs: (batch, num_classes) - xac suat sau softmax
            gate_scores: (batch, 1) - muc do su dung metadata
        """
        self.eval()
        # Tam bat training mode de lay gate_scores, roi tat lai
        phobert_out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = phobert_out.last_hidden_state

        text_vector = self.text_branch(embeddings, attention_mask)
        metadata_vector = self.metadata_branch(metadata)
        gated_metadata, gate_scores = self.metadata_gate(text_vector, metadata_vector)

        fused = torch.cat([text_vector, gated_metadata], dim=1)
        fused = self.layer_norm(fused)
        hidden = F.relu(self.fc(fused))
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        probs = torch.softmax(logits, dim=-1)
        return probs, gate_scores

    def phobert_parameters(self):
        """Tra ve tham so cua PhoBERT backbone."""
        return self.phobert.parameters()

    def head_parameters(self):
        """Tra ve tat ca tham so KHONG thuoc PhoBERT (CNN + metadata +
        gate + FC + classifier) - dung de gan learning rate rieng
        (discriminative fine-tuning)."""
        phobert_ids = {id(p) for p in self.phobert.parameters()}
        return [p for p in self.parameters() if id(p) not in phobert_ids]
