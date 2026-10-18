# -*- coding: utf-8 -*-
"""
tests/test_model.py
===================
Unit tests cho cac module kien truc cua PhoBERT-AMP v2:
  1. MultiKernelCNNBranch
  2. MetadataFieldEmbedding (3 truong metadata)
  3. CrossAttention
  4. AdaptiveMetadataGate
  5. Integration forward & backward pass
"""

import math
import os
import sys
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.model import (
    AdaptiveMetadataGate,
    CrossAttention,
    MetadataFieldEmbedding,
    MultiKernelCNNBranch,
    PhoBertAMP,
)


class TestMultiKernelCNNBranch:
    def test_output_shape(self):
        batch_size = 4
        seq_len = 32
        embed_dim = 768
        filter_sizes = [2, 3, 4, 5]
        num_filters = 64

        cnn = MultiKernelCNNBranch(
            embed_dim=embed_dim,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
        )
        x = torch.randn(batch_size, seq_len, embed_dim)
        mask = torch.ones(batch_size, seq_len)

        out = cnn(x, mask)
        assert out.shape == (batch_size, num_filters * len(filter_sizes))
        assert not torch.isnan(out).any()

    def test_short_sequence_handling(self):
        """Kiem tra chuoi ngan hon max filter size (vd seq_len=3, max kernel=5)."""
        cnn = MultiKernelCNNBranch(
            embed_dim=128,
            filter_sizes=[2, 3, 4, 5],
            num_filters=32,
        )
        x = torch.randn(2, 3, 128)
        mask = torch.ones(2, 3)

        out = cnn(x, mask)
        assert out.shape == (2, 32 * 4)
        assert not torch.isnan(out).any()


class TestMetadataFieldEmbedding:
    def test_flat_and_matrix_shapes(self):
        num_fields = 3
        per_field_dim = 32
        batch_size = 3

        embedder = MetadataFieldEmbedding(
            num_fields=num_fields,
            per_field_dim=per_field_dim,
        )
        metadata = torch.randn(batch_size, num_fields)

        # Flat output
        flat_out = embedder(metadata)
        assert flat_out.shape == (batch_size, num_fields * per_field_dim)

        # Matrix output for Cross-Attention
        matrix_out = embedder.forward_as_matrix(metadata)
        assert matrix_out.shape == (batch_size, num_fields, per_field_dim)

    def test_independent_embeddings(self):
        """Moi field phai duoc embed boi layer rieng biet."""
        embedder = MetadataFieldEmbedding(num_fields=3, per_field_dim=16)
        assert len(embedder.field_embeddings) == 3


class TestCrossAttention:
    def test_cross_attention_forward(self):
        batch_size = 4
        num_fields = 3
        text_dim = 256
        meta_field_dim = 32
        d_k = 32
        d_v = 32

        cross_attn = CrossAttention(
            text_dim=text_dim,
            meta_field_dim=meta_field_dim,
            d_k=d_k,
            d_v=d_v,
            dropout=0.0,
        )

        text_vec = torch.randn(batch_size, text_dim)
        meta_matrix = torch.randn(batch_size, num_fields, meta_field_dim)

        v_context, attn_weights = cross_attn(text_vec, meta_matrix)

        # Kiem tra shape
        assert v_context.shape == (batch_size, d_v)
        assert attn_weights.shape == (batch_size, num_fields)

        # Kiem tra tong attention weights = 1.0 moi sample
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)

        # Tat ca trong so phai >= 0
        assert (attn_weights >= 0).all()


class TestAdaptiveMetadataGate:
    def test_gate_bounds_and_shape(self):
        batch_size = 4
        text_dim = 256
        meta_dim = 32
        gate_max_val = 0.5

        gate = AdaptiveMetadataGate(
            text_dim=text_dim,
            metadata_dim=meta_dim,
            hidden_dim=64,
            gate_max_val=gate_max_val,
        )

        text_vec = torch.randn(batch_size, text_dim)
        meta_vec = torch.randn(batch_size, meta_dim)

        gated_meta, gate_scores = gate(text_vec, meta_vec)

        assert gated_meta.shape == (batch_size, meta_dim)
        assert gate_scores.shape == (batch_size, 1)

        # Gate score phai nam trong [0, gate_max_val]
        assert (gate_scores >= 0).all()
        assert (gate_scores <= gate_max_val).all()


class TestPhoBertAMPIntegration:
    @pytest.fixture
    def mock_model(self):
        """Tao model PhoBertAMP voi mock phobert backbone de test nhanh ma khong can tai weights."""
        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MagicMock()
                self.config.hidden_size = 768

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                batch_size, seq_len = input_ids.shape
                # Tra ve tensor gia lap output PhoBERT: (batch, seq_len, 768)
                last_hidden = torch.randn(batch_size, seq_len, 768, requires_grad=True)
                mock_out = MagicMock()
                mock_out.last_hidden_state = last_hidden
                return mock_out

        model = PhoBertAMP(
            phobert_model_name="vinai/phobert-base",
            num_classes=3,
            cnn_filter_sizes=[2, 3],
            cnn_num_filters=32,
            metadata_dim=3,
            per_field_embed_dim=16,
            cross_attention_dim=16,
            gate_hidden_dim=32,
            fc_hidden_dim=64,
        )
        # Thay backbone bang dummy
        model.phobert = DummyBackbone()
        return model

    def test_train_forward_backward(self, mock_model):
        mock_model.train()
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        metadata = torch.randn(batch_size, 3)

        logits, gate_scores, attn_weights = mock_model(input_ids, attention_mask, metadata)

        assert logits.shape == (batch_size, 3)
        assert gate_scores.shape == (batch_size, 1)
        assert attn_weights.shape == (batch_size, 3)

        # Test backward
        loss = logits.sum() + gate_scores.sum()
        loss.backward()

        # Kiem tra gradients
        for p in mock_model.head_parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_eval_forward(self, mock_model):
        mock_model.eval()
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        metadata = torch.randn(batch_size, 3)

        # Eval mode: forward tra ve duy nhat logits
        logits = mock_model(input_ids, attention_mask, metadata)
        assert logits.shape == (batch_size, 3)

        # predict_with_gate tra ve probs, gate_scores, attn_weights
        probs, gate_scores, attn_weights = mock_model.predict_with_gate(
            input_ids, attention_mask, metadata
        )
        assert probs.shape == (batch_size, 3)
        assert gate_scores.shape == (batch_size, 1)
        assert attn_weights.shape == (batch_size, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_aux_classifier_and_gate_init(self, mock_model):
        """Kiem tra khoi tao gate lanh manh va auxiliary classifier chong hoc vet metadata."""
        # 1. Kiem tra gate khoi tao: quanh 0.225 (= 0.5 * gate_max_val=0.45, khong phai 0)
        gate = AdaptiveMetadataGate(text_dim=64, metadata_dim=32, gate_max_val=0.45)
        sample_text = torch.zeros(4, 64)
        sample_meta = torch.zeros(4, 32)
        _, gate_score = gate(sample_text, sample_meta)
        # Sigmoid(0) = 0.5, nhan voi gate_max_val=0.45 -> 0.225
        assert torch.allclose(gate_score, torch.full_like(gate_score, 0.225), atol=0.05)

        # 2. Kiem tra PhoBertAMP voi use_aux_classifier=True
        mock_model.use_aux_classifier = True
        mock_model.aux_classifier = torch.nn.Linear(mock_model.text_branch.output_dim, mock_model.num_classes)
        mock_model.train()

        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16)
        metadata = torch.randn(2, 3)

        logits, gate_scores, attn_weights = mock_model(input_ids, attention_mask, metadata)
        assert mock_model.last_aux_logits is not None
        assert mock_model.last_aux_logits.shape == (2, 3)

        # Backward bao gom ca main logits va aux logits
        loss = logits.sum() + mock_model.last_aux_logits.sum() + gate_scores.sum()
        loss.backward()
        assert mock_model.aux_classifier.weight.grad is not None

