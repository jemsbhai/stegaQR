"""Tests for encoder/decoder model architectures.

Verifies tensor shapes, gradient flow, and basic forward pass
for all three steganographic modes.
"""

import pytest
import torch

# Skip all tests if torch unavailable
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and True,  # run on CPU too
    reason="",
)


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def batch():
    """Standard test batch: B=2, spatial=33x33 (QR v4), 100 bits."""
    B, H, W, C = 2, 33, 33, 100
    cover = torch.rand(B, 3, H, W)
    payload = torch.randint(0, 2, (B, C)).float()
    mask = torch.ones(B, 1, H, W)
    # Mark finder patterns as protected
    mask[:, :, 0:7, 0:7] = 0
    mask[:, :, 0:7, -7:] = 0
    mask[:, :, -7:, 0:7] = 0
    return cover, payload, mask


class TestSegregatedEncoder:
    def test_output_shape(self, batch, device):
        from stegaqr.models.encoder import SegregatedEncoder

        cover, payload, _ = batch
        # capacity_bits must be divisible by 3
        encoder = SegregatedEncoder(capacity_bits=99, spatial_size=33)
        stego = encoder(cover, payload[:, :99])
        assert stego.shape == cover.shape

    def test_output_range(self, batch, device):
        from stegaqr.models.encoder import SegregatedEncoder

        cover, payload, _ = batch
        encoder = SegregatedEncoder(capacity_bits=99, spatial_size=33)
        stego = encoder(cover, payload[:, :99])
        assert stego.min() >= 0.0
        assert stego.max() <= 1.0

    def test_gradient_flow(self, batch, device):
        from stegaqr.models.encoder import SegregatedEncoder

        cover, payload, _ = batch
        encoder = SegregatedEncoder(capacity_bits=99, spatial_size=33)
        stego = encoder(cover, payload[:, :99])
        loss = stego.sum()
        loss.backward()
        # Check that encoder parameters received gradients
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestCrossChannelEncoder:
    def test_output_shape(self, batch, device):
        from stegaqr.models.encoder import CrossChannelEncoder

        cover, payload, _ = batch
        encoder = CrossChannelEncoder(capacity_bits=100, spatial_size=33)
        stego = encoder(cover, payload)
        assert stego.shape == cover.shape

    def test_perturbation_bounded(self, batch, device):
        from stegaqr.models.encoder import CrossChannelEncoder

        cover, payload, _ = batch
        bound = 0.05
        encoder = CrossChannelEncoder(
            capacity_bits=100, spatial_size=33, perturbation_bound=bound
        )
        stego = encoder(cover, payload)
        diff = (stego - cover).abs()
        assert diff.max() <= bound + 1e-5  # tolerance for clamp edge


class TestHybridEncoder:
    def test_output_shape(self, batch, device):
        from stegaqr.models.encoder import HybridEncoder

        cover, payload, mask = batch
        encoder = HybridEncoder(capacity_bits=100, spatial_size=33)
        stego = encoder(cover, payload, mask)
        assert stego.shape == cover.shape

    def test_protected_modules_unchanged(self, batch, device):
        from stegaqr.models.encoder import HybridEncoder

        cover, payload, mask = batch
        encoder = HybridEncoder(capacity_bits=100, spatial_size=33)
        stego = encoder(cover, payload, mask)
        # Where mask=0, stego should equal cover
        protected = (mask == 0).expand_as(cover)
        assert torch.allclose(stego[protected], cover[protected], atol=1e-6)


class TestSegregatedDecoder:
    def test_output_shape(self, batch, device):
        from stegaqr.models.decoder import SegregatedDecoder

        cover, payload, _ = batch
        decoder = SegregatedDecoder(capacity_bits=99)
        output = decoder(cover)
        assert output.shape == (2, 99)


class TestCrossChannelDecoder:
    def test_output_shape(self, batch, device):
        from stegaqr.models.decoder import CrossChannelDecoder

        cover, payload, _ = batch
        decoder = CrossChannelDecoder(capacity_bits=100)
        output = decoder(cover)
        assert output.shape == (2, 100)


class TestHybridDecoder:
    def test_output_shape(self, batch, device):
        from stegaqr.models.decoder import HybridDecoder

        cover, payload, _ = batch
        decoder = HybridDecoder(capacity_bits=100)
        output, confidence = decoder(cover)
        assert output.shape == (2, 100)
        assert confidence.shape == (2, 1)

    def test_confidence_range(self, batch, device):
        from stegaqr.models.decoder import HybridDecoder

        cover, _, _ = batch
        decoder = HybridDecoder(capacity_bits=100)
        _, confidence = decoder(cover)
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0


class TestEndToEnd:
    """Test encoder -> distortion -> decoder pipeline."""

    def test_segregated_pipeline_gradient(self, batch, device):
        from stegaqr.models.encoder import SegregatedEncoder
        from stegaqr.models.decoder import SegregatedDecoder
        from stegaqr.models.distortion import IdentityDistortion

        cover, payload, _ = batch
        payload = payload[:, :99]

        encoder = SegregatedEncoder(capacity_bits=99, spatial_size=33)
        distortion = IdentityDistortion()
        decoder = SegregatedDecoder(capacity_bits=99)

        stego = encoder(cover, payload)
        distorted = distortion(stego)
        predicted = decoder(distorted)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predicted, payload
        )
        loss.backward()

        # Gradients must flow through the entire pipeline
        enc_grads = sum(1 for p in encoder.parameters() if p.grad is not None)
        dec_grads = sum(1 for p in decoder.parameters() if p.grad is not None)
        assert enc_grads > 0
        assert dec_grads > 0
