"""Tests for encoder/decoder model architectures (v2).

Verifies tensor shapes, gradient flow, and basic forward pass
for all three steganographic modes.
"""

import pytest
import torch


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def batch():
    """Standard test batch: B=2, 33x33 modules at module_size=1, 100 bits."""
    B, H, W, C = 2, 33, 33, 100
    cover = torch.rand(B, 3, H, W)
    payload = torch.randint(0, 2, (B, C)).float()
    mask = torch.ones(B, 1, H, W)
    mask[:, :, 0:7, 0:7] = 0
    mask[:, :, 0:7, -7:] = 0
    mask[:, :, -7:, 0:7] = 0
    return cover, payload, mask


@pytest.fixture
def scaled_batch():
    """Scaled batch: B=2, 33x33 modules at module_size=4 = 132x132."""
    B, H, W, C = 2, 132, 132, 100
    cover = torch.rand(B, 3, H, W)
    payload = torch.randint(0, 2, (B, C)).float()
    mask = torch.ones(B, 1, H, W)
    # Finder patterns at module_size=4
    mask[:, :, 0:28, 0:28] = 0
    mask[:, :, 0:28, -28:] = 0
    mask[:, :, -28:, 0:28] = 0
    return cover, payload, mask


class TestSegregatedEncoder:
    def test_output_shape(self, batch):
        from stegaqr.models.encoder import SegregatedEncoder
        cover, payload, _ = batch
        encoder = SegregatedEncoder(capacity_bits=99)
        stego = encoder(cover, payload[:, :99])
        assert stego.shape == cover.shape

    def test_output_range(self, batch):
        from stegaqr.models.encoder import SegregatedEncoder
        cover, payload, _ = batch
        encoder = SegregatedEncoder(capacity_bits=99)
        stego = encoder(cover, payload[:, :99])
        assert stego.min() >= 0.0
        assert stego.max() <= 1.0

    def test_gradient_flow(self, batch):
        from stegaqr.models.encoder import SegregatedEncoder
        cover, payload, _ = batch
        encoder = SegregatedEncoder(capacity_bits=99)
        stego = encoder(cover, payload[:, :99])
        loss = stego.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_scaled_input(self, scaled_batch):
        from stegaqr.models.encoder import SegregatedEncoder
        cover, payload, _ = scaled_batch
        encoder = SegregatedEncoder(capacity_bits=99)
        stego = encoder(cover, payload[:, :99])
        assert stego.shape == cover.shape


class TestCrossChannelEncoder:
    def test_output_shape(self, batch):
        from stegaqr.models.encoder import CrossChannelEncoder
        cover, payload, _ = batch
        encoder = CrossChannelEncoder(capacity_bits=100)
        stego = encoder(cover, payload)
        assert stego.shape == cover.shape

    def test_perturbation_bounded(self, batch):
        from stegaqr.models.encoder import CrossChannelEncoder
        cover, payload, _ = batch
        bound = 0.3
        encoder = CrossChannelEncoder(capacity_bits=100, perturbation_bound=bound)
        stego = encoder(cover, payload)
        diff = (stego - cover).abs()
        assert diff.max() <= bound + 1e-5


class TestHybridEncoder:
    def test_output_shape(self, batch):
        from stegaqr.models.encoder import HybridEncoder
        cover, payload, mask = batch
        encoder = HybridEncoder(capacity_bits=100)
        stego = encoder(cover, payload, mask)
        assert stego.shape == cover.shape

    def test_protected_modules_unchanged(self, batch):
        from stegaqr.models.encoder import HybridEncoder
        cover, payload, mask = batch
        encoder = HybridEncoder(capacity_bits=100)
        stego = encoder(cover, payload, mask)
        protected = (mask == 0).expand_as(cover)
        assert torch.allclose(stego[protected], cover[protected], atol=1e-6)


class TestSegregatedDecoder:
    def test_output_shape(self, batch):
        from stegaqr.models.decoder import SegregatedDecoder
        cover, _, _ = batch
        decoder = SegregatedDecoder(capacity_bits=99)
        output = decoder(cover)
        assert output.shape == (2, 99)


class TestCrossChannelDecoder:
    def test_output_shape(self, batch):
        from stegaqr.models.decoder import CrossChannelDecoder
        cover, _, _ = batch
        decoder = CrossChannelDecoder(capacity_bits=100)
        output = decoder(cover)
        assert output.shape == (2, 100)


class TestHybridDecoder:
    def test_output_shape(self, batch):
        from stegaqr.models.decoder import HybridDecoder
        cover, _, _ = batch
        decoder = HybridDecoder(capacity_bits=100)
        output, confidence = decoder(cover)
        assert output.shape == (2, 100)
        assert confidence.shape == (2, 1)

    def test_confidence_range(self, batch):
        from stegaqr.models.decoder import HybridDecoder
        cover, _, _ = batch
        decoder = HybridDecoder(capacity_bits=100)
        _, confidence = decoder(cover)
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0


class TestEndToEnd:
    """Test encoder -> distortion -> decoder pipeline."""

    def test_segregated_pipeline_gradient(self, batch):
        from stegaqr.models.encoder import SegregatedEncoder
        from stegaqr.models.decoder import SegregatedDecoder
        from stegaqr.models.distortion import IdentityDistortion

        cover, payload, _ = batch
        payload = payload[:, :99]

        encoder = SegregatedEncoder(capacity_bits=99)
        distortion = IdentityDistortion()
        decoder = SegregatedDecoder(capacity_bits=99)

        stego = encoder(cover, payload)
        distorted = distortion(stego)
        predicted = decoder(distorted)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, payload)
        loss.backward()

        enc_grads = sum(1 for p in encoder.parameters() if p.grad is not None)
        dec_grads = sum(1 for p in decoder.parameters() if p.grad is not None)
        assert enc_grads > 0
        assert dec_grads > 0

    def test_scaled_pipeline(self, scaled_batch):
        """Full pipeline at scaled resolution."""
        from stegaqr.models.encoder import CrossChannelEncoder
        from stegaqr.models.decoder import CrossChannelDecoder
        from stegaqr.models.distortion import IdentityDistortion

        cover, payload, _ = scaled_batch
        encoder = CrossChannelEncoder(capacity_bits=100)
        decoder = CrossChannelDecoder(capacity_bits=100)
        distortion = IdentityDistortion()

        stego = encoder(cover, payload)
        distorted = distortion(stego)
        predicted = decoder(distorted)

        assert predicted.shape == (2, 100)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, payload)
        loss.backward()
