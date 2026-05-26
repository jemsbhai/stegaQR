"""Loss functions for StegaQR training.

Combined loss balances four objectives:
  1. Hidden payload recovery accuracy (decode loss)
  2. Visual imperceptibility (perceptual loss)
  3. Public QR decodability preservation (decodability loss)
  4. Calibrated confidence estimation (confidence loss, hybrid only)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecodeLoss(nn.Module):
    """Binary cross-entropy loss for hidden payload recovery.

    Primary training objective: the decoder must recover the hidden bits.
    """

    def forward(
        self, predicted_logits: torch.Tensor, target_bits: torch.Tensor
    ) -> torch.Tensor:
        """
        predicted_logits: (B, C) raw logits from decoder
        target_bits: (B, C) ground truth bits in {0, 1}
        """
        return F.binary_cross_entropy_with_logits(predicted_logits, target_bits)


class PerceptualLoss(nn.Module):
    """Perceptual quality loss — stego image must look like the cover.

    Combines MSE (pixel-level fidelity) with a structural term that
    penalizes large per-pixel deviations more than small ones.

    The perturbation_bound in the encoder already hard-clamps the
    maximum deviation; this loss provides a soft gradient signal
    encouraging the encoder to use minimal perturbation.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        """
        alpha : weight for MSE component
        beta : weight for max-deviation penalty
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, stego: torch.Tensor, cover: torch.Tensor
    ) -> torch.Tensor:
        """
        stego: (B, 3, H, W) stego image
        cover: (B, 3, H, W) cover image
        """
        # MSE: average pixel distortion
        mse = F.mse_loss(stego, cover)

        # Max deviation penalty: penalize outlier perturbations
        diff = (stego - cover).abs()
        max_dev = diff.amax(dim=(1, 2, 3)).mean()  # mean of per-sample max

        return self.alpha * mse + self.beta * max_dev


class DecodabilityLoss(nn.Module):
    """Soft proxy for QR decodability preservation.

    We can't run pyzbar in the training loop (non-differentiable), so
    this loss uses a differentiable proxy: the stego image's module
    values at finder/timing pattern positions must remain close to
    their original binary values (0 or 1).

    For hybrid mode, the encoder's mask already zeros perturbation on
    protected modules. This loss provides additional gradient signal
    for data modules whose color changes might indirectly affect
    a QR reader's global thresholding step.

    Specifically: modules that should be black (cover=0) must stay
    below 0.3, and modules that should be white (cover=1) must stay
    above 0.7. Violations are penalized with a hinge loss.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin  # allowed deviation from 0 or 1

    def forward(
        self, stego: torch.Tensor, cover: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        stego: (B, 3, H, W)
        cover: (B, 3, H, W)
        mask: (B, 1, H, W) — 0=protected, 1=data
        """
        # Focus on protected modules (mask=0)
        protected = (1.0 - mask).expand_as(stego)

        # Grayscale of stego at protected positions
        stego_gray = stego.mean(dim=1, keepdim=True).expand_as(stego)
        cover_gray = cover.mean(dim=1, keepdim=True).expand_as(cover)

        # Only look at protected modules
        stego_protected = stego_gray * protected
        cover_protected = cover_gray * protected

        # Hinge loss: protected modules must stay within margin of original
        deviation = (stego_protected - cover_protected).abs()
        violation = F.relu(deviation - self.margin)

        # Normalize by number of protected modules
        num_protected = protected.sum().clamp(min=1.0)
        return violation.sum() / num_protected


class ConfidenceLoss(nn.Module):
    """Expected Calibration Error (ECE) loss for confidence estimation.

    Only used with HybridDecoder. The confidence output should predict
    the probability that the hidden payload was correctly decoded.

    We approximate this with a binary target: confidence should be high
    when bit accuracy is high, and low when it's low.
    """

    def forward(
        self,
        confidence: torch.Tensor,
        predicted_logits: torch.Tensor,
        target_bits: torch.Tensor,
    ) -> torch.Tensor:
        """
        confidence: (B, 1) predicted confidence in [0, 1]
        predicted_logits: (B, C) decoder output logits
        target_bits: (B, C) ground truth bits
        """
        # Compute per-sample bit accuracy as the "true" confidence
        with torch.no_grad():
            predicted_bits = (torch.sigmoid(predicted_logits) > 0.5).float()
            accuracy = (predicted_bits == target_bits).float().mean(dim=1, keepdim=True)

        # BCE between predicted confidence and actual accuracy
        return F.binary_cross_entropy(confidence, accuracy)


class StegaQRLoss(nn.Module):
    """Combined loss for end-to-end StegaQR training.

    L_total = λ_decode * L_decode
            + λ_perceptual * L_perceptual
            + λ_decodability * L_decodability
            + λ_confidence * L_confidence  (hybrid mode only)

    Parameters
    ----------
    lambda_decode : float
        Weight for hidden payload recovery loss.
    lambda_perceptual : float
        Weight for visual quality loss.
    lambda_decodability : float
        Weight for QR decodability preservation.
    lambda_confidence : float
        Weight for confidence calibration (hybrid only).
    use_confidence : bool
        Whether to include confidence loss (True for hybrid mode).
    """

    def __init__(
        self,
        lambda_decode: float = 1.0,
        lambda_perceptual: float = 1.0,
        lambda_decodability: float = 0.5,
        lambda_confidence: float = 0.1,
        use_confidence: bool = False,
    ):
        super().__init__()
        self.lambda_decode = lambda_decode
        self.lambda_perceptual = lambda_perceptual
        self.lambda_decodability = lambda_decodability
        self.lambda_confidence = lambda_confidence
        self.use_confidence = use_confidence

        self.decode_loss = DecodeLoss()
        self.perceptual_loss = PerceptualLoss()
        self.decodability_loss = DecodabilityLoss()
        self.confidence_loss = ConfidenceLoss()

    def forward(
        self,
        predicted_logits: torch.Tensor,
        target_bits: torch.Tensor,
        stego: torch.Tensor,
        cover: torch.Tensor,
        mask: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components and total.

        Returns
        -------
        dict with keys:
            'total': combined weighted loss (for backprop)
            'decode': payload recovery loss
            'perceptual': visual quality loss
            'decodability': QR preservation loss
            'confidence': confidence calibration loss (if applicable)
            'bit_accuracy': per-bit accuracy (metric, not a loss)
        """
        l_decode = self.decode_loss(predicted_logits, target_bits)
        l_perceptual = self.perceptual_loss(stego, cover)
        l_decodability = self.decodability_loss(stego, cover, mask)

        total = (
            self.lambda_decode * l_decode
            + self.lambda_perceptual * l_perceptual
            + self.lambda_decodability * l_decodability
        )

        losses = {
            "total": total,
            "decode": l_decode.detach(),
            "perceptual": l_perceptual.detach(),
            "decodability": l_decodability.detach(),
        }

        if self.use_confidence and confidence is not None:
            l_confidence = self.confidence_loss(
                confidence, predicted_logits, target_bits
            )
            total = total + self.lambda_confidence * l_confidence
            losses["total"] = total
            losses["confidence"] = l_confidence.detach()

        # Compute bit accuracy as a monitoring metric
        with torch.no_grad():
            pred_bits = (torch.sigmoid(predicted_logits) > 0.5).float()
            bit_acc = (pred_bits == target_bits).float().mean()
            losses["bit_accuracy"] = bit_acc

        return losses
