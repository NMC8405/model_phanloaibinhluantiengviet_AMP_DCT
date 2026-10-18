# -*- coding: utf-8 -*-
"""
model.py
========
Kien truc mo hinh PhoBERT-AMP v2 (Adaptive Metadata-aware PhoBERT voi
Cross-Attention):

  Nhanh van ban:
    PhoBERT (Contextual Embedding) -> Multi-Kernel CNN (Conv1D + BatchNorm
    + ReLU) -> Global Max Pooling -> Vector van ban (v_text)

  Nhanh metadata (Per-Field Embedding):
    Moi truong metadata (comment_length, word_count,
    has_image - da loai bo rating va quality de tranh Target Leakage) -> Embedding rieng
    (Dense + ReLU) -> Ma tran metadata M_meta (num_fields x per_field_dim)

  Cross-Attention (MOI):
    Text Feature tao Query: Q = v_text @ W_Q
    Metadata tao Key va Value: K = M_meta @ W_K, V = M_meta @ W_V
    Attention Weights: A = Softmax(Q @ K^T / sqrt(d_k))
    Contextual Metadata: v_meta_context = A @ V
    => Dac trung metadata da duoc dieu chinh theo noi dung van ban.

  Adaptive Metadata Gate:
    concat(text_vec, v_meta_context) -> MLP 2 lop -> Sigmoid -> gate_score
    gated_metadata = gate_score * v_meta_context
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

import math
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

    Moi truong metadata (comment_length, word_count,
    has_image) co RIENG 1 embedding layer
    (1 -> per_field_dim). Ho tro 2 che do output:
      - forward(): concat tat ca -> (batch, num_fields * per_field_dim) [flat]
      - forward_as_matrix(): -> (batch, num_fields, per_field_dim) [matrix]
        Dung cho Cross-Attention (moi field la 1 "token" metadata).

    Loi ich: model hoc bieu dien RIENG cho tung truong, khong bi "tron lan"
    ngay tu dau. Vi du, "rating cao" va "chat luong tot" duoc encode
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
            (batch, num_fields * per_field_dim) - concat flat
        """
        field_outputs = []
        for i, embed in enumerate(self.field_embeddings):
            field_input = metadata[:, i:i+1]  # (batch, 1)
            field_outputs.append(embed(field_input))  # (batch, per_field_dim)
        return torch.cat(field_outputs, dim=1)  # (batch, num_fields * per_field_dim)

    def forward_as_matrix(self, metadata: torch.Tensor) -> torch.Tensor:
        """Tra ve output dang ma tran (batch, num_fields, per_field_dim).

        Dung cho Cross-Attention: moi field metadata la 1 "token" rieng,
        tuong tu cach NLP coi moi tu la 1 token trong sequence.

        Args:
            metadata: (batch, num_fields) - vector metadata da chuan hoa
        Returns:
            (batch, num_fields, per_field_dim)
        """
        field_outputs = []
        for i, embed in enumerate(self.field_embeddings):
            field_input = metadata[:, i:i+1]  # (batch, 1)
            field_outputs.append(embed(field_input))  # (batch, per_field_dim)
        # Stack thay vi concat de giu rieng tung field
        return torch.stack(field_outputs, dim=1)  # (batch, num_fields, per_field_dim)


class CrossAttention(nn.Module):
    """Cross-Attention giua Text Features va Metadata Fields.

    Giup mo hinh xac dinh duoc metadata nao co muc do lien quan cao hon
    doi voi noi dung binh luan cu the.

    Vi du thuc te:
      - Binh luan "San pham tot nhung gia cao" + rating=3, quality=5
        => Cross-Attention cho weight cao hon cho `rating` (text de cap
        den gia ca -> lien quan den danh gia tong the)
      - Binh luan "Chat lieu mem mai, dung ben" + rating=5, quality=4
        => Cross-Attention cho weight cao hon cho `quality` (text de cap
        den chat lieu -> lien quan den quality)

    Cong thuc:
        Q = v_text @ W_Q + b_Q         (batch, 1, d_k)
        K = M_meta @ W_K + b_K         (batch, num_fields, d_k)
        V = M_meta @ W_V + b_V         (batch, num_fields, d_v)
        A_score = Q @ K^T / sqrt(d_k)  (batch, 1, num_fields)
        A = Softmax(A_score)            (batch, 1, num_fields)
        v_meta_context = A @ V          (batch, 1, d_v) -> squeeze -> (batch, d_v)
    """

    def __init__(
        self,
        text_dim: int,
        meta_field_dim: int,
        d_k: int = 32,
        d_v: int = 32,
        dropout: float = 0.1,
    ):
        """
        Args:
            text_dim: chieu cua text vector (= CNN output dim, vd 512)
            meta_field_dim: chieu cua moi metadata field embedding (= per_field_dim, vd 32)
            d_k: chieu cua Key/Query space
            d_v: chieu cua Value space (= chieu output cua cross-attention)
            dropout: dropout tren attention weights
        """
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v

        # Projection layers
        self.W_Q = nn.Linear(text_dim, d_k)       # Text -> Query
        self.W_K = nn.Linear(meta_field_dim, d_k)  # Metadata -> Key
        self.W_V = nn.Linear(meta_field_dim, d_v)  # Metadata -> Value

        self.attn_dropout = nn.Dropout(dropout)

        # LayerNorm cho output de on dinh training
        self.layer_norm = nn.LayerNorm(d_v)

        self.output_dim = d_v

    def forward(
        self,
        text_vector: torch.Tensor,
        metadata_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_vector: (batch, text_dim) - dac trung van ban tu CNN
            metadata_matrix: (batch, num_fields, meta_field_dim) - ma tran
                metadata tu Per-Field Embedding (moi field la 1 "token")
        Returns:
            v_meta_context: (batch, d_v) - dac trung metadata da duoc dieu
                chinh theo noi dung van ban
            attention_weights: (batch, num_fields) - trong so attention cho
                moi metadata field (tong = 1, dung de debug/visualize)
        """
        # Query tu text: (batch, text_dim) -> (batch, 1, d_k)
        Q = self.W_Q(text_vector).unsqueeze(1)  # (batch, 1, d_k)

        # Key va Value tu metadata: (batch, num_fields, meta_field_dim) ->
        K = self.W_K(metadata_matrix)  # (batch, num_fields, d_k)
        V = self.W_V(metadata_matrix)  # (batch, num_fields, d_v)

        # Scaled dot-product attention
        # A_score = Q @ K^T / sqrt(d_k) -> (batch, 1, num_fields)
        scale = math.sqrt(self.d_k)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / scale  # (batch, 1, num_fields)

        # Softmax de chuan hoa thanh weights (tong = 1)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, num_fields)
        attn_weights = self.attn_dropout(attn_weights)

        # Contextual metadata = weighted sum cua Value vectors
        # (batch, 1, num_fields) @ (batch, num_fields, d_v) -> (batch, 1, d_v)
        v_meta_context = torch.bmm(attn_weights, V).squeeze(1)  # (batch, d_v)

        # LayerNorm de on dinh
        v_meta_context = self.layer_norm(v_meta_context)

        # Tra ve attention weights dang (batch, num_fields) de debug
        attn_weights_flat = attn_weights.squeeze(1)  # (batch, num_fields)

        return v_meta_context, attn_weights_flat


class AdaptiveMetadataGate(nn.Module):
    """Adaptive Metadata Gate - co che hoc trong so metadata.

    Gate quyet dinh MUC DO su dung metadata cho TUNG SAMPLE (khong phai
    toan cuc). Dieu nay cho phep model:
      - Khi text DA RO RANG (vd "san pham rat te"): gate ~ 0, bo qua metadata
      - Khi text MO HO (vd "ok"): gate ~ 1, su dung metadata (rating, quality)
        de ho tro phan loai

    Trong phien ban v2 voi Cross-Attention, gate nhan v_meta_context (da duoc
    dieu chinh theo noi dung van ban) thay vi raw metadata vector.

    Cong thuc:
        gate_input = concat(text_vector, v_meta_context)
        gate_score = sigmoid(W2 * ReLU(W1 * gate_input + b1) + b2) * gate_max_val
        gated_metadata = gate_score * v_meta_context

    gate_score la scalar trong (0, gate_max_val) cho moi sample trong batch.
    """

    def __init__(self, text_dim: int, metadata_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.2, gate_max_val: float = 0.45):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(text_dim + metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Khoi tao bias = 0 de ban dau sigmoid cho gia tri ~0.5, tranh gate collapse som
        nn.init.constant_(self.gate_net[-2].bias, 0.0)
        self.gate_max_val = gate_max_val

    def forward(self, text_vector: torch.Tensor, metadata_vector: torch.Tensor):
        """
        Args:
            text_vector: (batch, text_dim)
            metadata_vector: (batch, metadata_dim) - v_meta_context tu Cross-Attention
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
    """PhoBERT-AMP v2: Adaptive Metadata-aware PhoBERT voi Cross-Attention.

    Ket hop 4 module chinh:
      1. Multi-Kernel CNN Branch: trich xuat dac trung van ban tu PhoBERT
      2. Per-Field Metadata Embedding: embed tung truong metadata rieng
      3. Cross-Attention (MOI): xac dinh metadata nao lien quan hon voi
         noi dung van ban cu the
      4. Adaptive Metadata Gate: hoc trong so de quyet dinh muc do dung
         metadata (da duoc dieu chinh boi Cross-Attention)

    Args:
        phobert_model_name: ten model tren HuggingFace Hub
        num_classes: so nhan dau ra (3: negative/normal/positive)
        cnn_filter_sizes, cnn_num_filters: cau hinh nhanh Multi-Kernel CNN
        metadata_dim: so truong metadata dau vao (3: comment_length, word_count, has_image)
        per_field_embed_dim: so chieu embed cho MOI truong metadata
        cross_attention_dim: d_k = d_v cho Cross-Attention
        cross_attention_dropout: dropout tren attention weights
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
        cross_attention_dim: int = 32,
        cross_attention_dropout: float = 0.1,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.2,
        fc_hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_phobert: bool = False,
        gate_max_val: float = 0.45,
        metadata_dropout: float = 0.40,
        use_aux_classifier: bool = False,
    ):
        super().__init__()
        self.metadata_dropout = metadata_dropout
        self.use_aux_classifier = use_aux_classifier

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

        # --- Cross-Attention (MOI): Text x Metadata ---
        self.cross_attention = CrossAttention(
            text_dim=self.text_branch.output_dim,
            meta_field_dim=per_field_embed_dim,
            d_k=cross_attention_dim,
            d_v=cross_attention_dim,
            dropout=cross_attention_dropout,
        )

        # --- Adaptive Metadata Gate ---
        # Nhan v_meta_context tu Cross-Attention (d_v dim) thay vi raw
        # metadata concat (num_fields * per_field_dim)
        cross_attn_output_dim = self.cross_attention.output_dim
        self.metadata_gate = AdaptiveMetadataGate(
            text_dim=self.text_branch.output_dim,
            metadata_dim=cross_attn_output_dim,
            hidden_dim=gate_hidden_dim,
            dropout=gate_dropout,
            gate_max_val=gate_max_val,
        )

        # --- Fusion head: LayerNorm + FC + Dropout + Classifier ---
        # fusion_dim = text_dim + cross_attention_output_dim
        fusion_dim = self.text_branch.output_dim + cross_attn_output_dim
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.fc = nn.Linear(fusion_dim, fc_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc_hidden_dim, num_classes)

        # --- Auxiliary Text Classifier (chong shortcut learning / hoc vet metadata) ---
        if use_aux_classifier:
            self.aux_classifier = nn.Linear(self.text_branch.output_dim, num_classes)
        else:
            self.aux_classifier = None
        self.last_aux_logits = None

        self.num_classes = num_classes

    def _forward_core(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core forward pass dung chung cho ca training va inference.

        Returns:
            logits: (batch, num_classes)
            gate_scores: (batch, 1)
            attention_weights: (batch, num_fields)
        """
        # --- PhoBERT encoding ---
        phobert_out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = phobert_out.last_hidden_state  # (batch, seq_len, hidden)

        # --- Nhanh van ban ---
        text_vector = self.text_branch(embeddings, attention_mask)  # (batch, cnn_out)

        # Auxiliary classifier chi dua tren van ban de bat buoc PhoBERT+CNN phai hoc ngu nghia
        if self.use_aux_classifier and self.aux_classifier is not None:
            self.last_aux_logits = self.aux_classifier(text_vector)
        else:
            self.last_aux_logits = None

        # --- Nhanh metadata ---
        # Dung forward_as_matrix() de co (batch, num_fields, per_field_dim)
        metadata_matrix = self.metadata_branch.forward_as_matrix(metadata)

        # Apply metadata dropout during training to prevent shortcut learning
        if self.training and self.metadata_dropout > 0:
            mask = (torch.rand(metadata_matrix.size(0), 1, 1, device=metadata_matrix.device) >= self.metadata_dropout).float()
            metadata_matrix = metadata_matrix * mask

        # --- Cross-Attention: Text x Metadata ---
        # v_meta_context la dac trung metadata da duoc dieu chinh theo van ban
        v_meta_context, attention_weights = self.cross_attention(
            text_vector, metadata_matrix
        )  # (batch, d_v), (batch, num_fields)

        # --- Adaptive Metadata Gate ---
        gated_metadata, gate_scores = self.metadata_gate(
            text_vector, v_meta_context
        )  # (batch, d_v), (batch, 1)

        # --- Fusion ---
        fused = torch.cat([text_vector, gated_metadata], dim=1)
        fused = self.layer_norm(fused)
        hidden = F.relu(self.fc(fused))
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        return logits, gate_scores, attention_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns:
            Khi training: (logits, gate_scores, attention_weights)
                - logits: (batch, num_classes) - CHUA qua softmax
                - gate_scores: (batch, 1) - trong so gate (de regularize)
                - attention_weights: (batch, num_fields) - trong so
                  cross-attention cho moi metadata field
            Khi eval: logits only (tuong thich nguoc voi pipeline cu)
        """
        logits, gate_scores, attention_weights = self._forward_core(
            input_ids, attention_mask, metadata
        )

        if self.training:
            return logits, gate_scores, attention_weights
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict + tra ve gate_score va attention weights (dung cho API/debug).

        Returns:
            probs: (batch, num_classes) - xac suat sau softmax
            gate_scores: (batch, 1) - muc do su dung metadata
            attention_weights: (batch, num_fields) - trong so cross-attention
                cho moi metadata field (comment_length, word_count, has_image)
        """
        self.eval()
        logits, gate_scores, attention_weights = self._forward_core(
            input_ids, attention_mask, metadata
        )
        probs = torch.softmax(logits, dim=-1)
        return probs, gate_scores, attention_weights

    def phobert_parameters(self):
        """Tra ve tham so cua PhoBERT backbone."""
        return self.phobert.parameters()

    def head_parameters(self):
        """Tra ve tat ca tham so KHONG thuoc PhoBERT (CNN + metadata +
        cross-attention + gate + FC + classifier) - dung de gan learning
        rate rieng (discriminative fine-tuning)."""
        phobert_ids = {id(p) for p in self.phobert.parameters()}
        return [p for p in self.parameters() if id(p) not in phobert_ids]
