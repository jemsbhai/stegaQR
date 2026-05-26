"""Training loop for StegaQR encoder-decoder pairs.

Implements the end-to-end pipeline:
  Encoder(cover, payload, [mask]) → stego
  Distortion(stego) → distorted
  Decoder(distorted) → predicted_logits
  Loss(predicted, target, stego, cover, mask) → backprop

Follows lab-runner protocol: checkpointing, seed management,
environment snapshots, and structured logging.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from stegaqr.data.synthetic import create_dataloaders
from stegaqr.models.distortion import DifferentiableDistortion, IdentityDistortion
from stegaqr.models.losses import StegaQRLoss
from stegaqr.utils.seed import set_all_seeds, save_seeds


def _get_encoder_decoder(mode: str, capacity_bits: int, spatial_size: int, device: str):
    """Instantiate the correct encoder-decoder pair for a given mode."""
    if mode == "segregated":
        from stegaqr.models.encoder import SegregatedEncoder
        from stegaqr.models.decoder import SegregatedDecoder

        # Capacity must be divisible by 3 for segregated mode
        capacity_bits = (capacity_bits // 3) * 3
        encoder = SegregatedEncoder(
            capacity_bits=capacity_bits, spatial_size=spatial_size
        ).to(device)
        decoder = SegregatedDecoder(capacity_bits=capacity_bits).to(device)
        return encoder, decoder, capacity_bits, False

    elif mode == "cross_channel":
        from stegaqr.models.encoder import CrossChannelEncoder
        from stegaqr.models.decoder import CrossChannelDecoder

        encoder = CrossChannelEncoder(
            capacity_bits=capacity_bits, spatial_size=spatial_size
        ).to(device)
        decoder = CrossChannelDecoder(capacity_bits=capacity_bits).to(device)
        return encoder, decoder, capacity_bits, False

    elif mode == "hybrid":
        from stegaqr.models.encoder import HybridEncoder
        from stegaqr.models.decoder import HybridDecoder

        encoder = HybridEncoder(
            capacity_bits=capacity_bits, spatial_size=spatial_size
        ).to(device)
        decoder = HybridDecoder(capacity_bits=capacity_bits).to(device)
        return encoder, decoder, capacity_bits, True

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'segregated', 'cross_channel', or 'hybrid'.")


def _count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    mode: str = "hybrid",
    capacity_bits: int = 100,
    qr_version: int = 4,
    ec_level: str = "M",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_train: int = 5000,
    num_val: int = 750,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "experiments/default",
    checkpoint_every: int = 10,
    lambda_decode: float = 1.0,
    lambda_perceptual: float = 1.0,
    lambda_decodability: float = 0.5,
    lambda_confidence: float = 0.1,
    use_distortion: bool = True,
) -> dict:
    """Train a StegaQR encoder-decoder pair.

    Parameters
    ----------
    mode : str
        'segregated', 'cross_channel', or 'hybrid'.
    capacity_bits : int
        Hidden payload size in bits.
    qr_version : int
        QR version (determines spatial size).
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    seed : int
        Random seed for reproducibility.
    device : str
        'cuda' or 'cpu'.
    output_dir : str
        Directory for checkpoints and logs.
    use_distortion : bool
        Whether to apply distortion layer during training.

    Returns
    -------
    dict with training history and final metrics.
    """
    # --- Setup ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)

    # Seed management
    seeds = set_all_seeds(seed)
    save_seeds(seeds, output_path / "seed.json")

    # Device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    spatial_size = 4 * qr_version + 17

    # --- Models ---
    encoder, decoder, capacity_bits, use_confidence = _get_encoder_decoder(
        mode, capacity_bits, spatial_size, device
    )

    print(f"Mode: {mode}")
    print(f"Capacity: {capacity_bits} bits")
    print(f"Spatial size: {spatial_size}x{spatial_size}")
    print(f"Encoder params: {_count_parameters(encoder):,}")
    print(f"Decoder params: {_count_parameters(decoder):,}")
    print(f"Device: {device}")

    # Distortion layer
    if use_distortion:
        distortion = DifferentiableDistortion().to(device)
    else:
        distortion = IdentityDistortion().to(device)

    # Loss
    criterion = StegaQRLoss(
        lambda_decode=lambda_decode,
        lambda_perceptual=lambda_perceptual,
        lambda_decodability=lambda_decodability,
        lambda_confidence=lambda_confidence,
        use_confidence=use_confidence,
    )

    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Data ---
    train_loader, val_loader = create_dataloaders(
        num_train=num_train,
        num_val=num_val,
        capacity_bits=capacity_bits,
        qr_version=qr_version,
        ec_level=ec_level,
        batch_size=batch_size,
    )

    # --- Save config ---
    config = {
        "mode": mode,
        "capacity_bits": capacity_bits,
        "qr_version": qr_version,
        "ec_level": ec_level,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_train": num_train,
        "num_val": num_val,
        "seed": seed,
        "device": device,
        "use_distortion": use_distortion,
        "encoder_params": _count_parameters(encoder),
        "decoder_params": _count_parameters(decoder),
        "lambda_decode": lambda_decode,
        "lambda_perceptual": lambda_perceptual,
        "lambda_decodability": lambda_decodability,
        "lambda_confidence": lambda_confidence,
        "spatial_size": spatial_size,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Training loop ---
    history = {
        "train_loss": [],
        "train_bit_acc": [],
        "val_loss": [],
        "val_bit_acc": [],
        "lr": [],
    }
    best_val_acc = 0.0

    for epoch in range(epochs):
        t0 = time.time()

        # ---- Train ----
        encoder.train()
        decoder.train()
        epoch_losses = []
        epoch_accs = []

        for batch in train_loader:
            cover = batch["cover"].to(device)
            payload = batch["payload"].to(device)
            mask = batch["mask"].to(device)

            # Forward: encoder -> distortion -> decoder
            if mode == "hybrid":
                stego = encoder(cover, payload, mask)
            else:
                stego = encoder(cover, payload)

            distorted = distortion(stego)

            if mode == "hybrid":
                predicted_logits, confidence = decoder(distorted)
            else:
                predicted_logits = decoder(distorted)
                confidence = None

            # Loss
            losses = criterion(
                predicted_logits, payload, stego, cover, mask, confidence
            )

            # Backprop
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_losses.append(losses["total"].item())
            epoch_accs.append(losses["bit_accuracy"].item())

        scheduler.step()

        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_acc = sum(epoch_accs) / len(epoch_accs)

        # ---- Validate ----
        encoder.eval()
        decoder.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for batch in val_loader:
                cover = batch["cover"].to(device)
                payload = batch["payload"].to(device)
                mask = batch["mask"].to(device)

                if mode == "hybrid":
                    stego = encoder(cover, payload, mask)
                else:
                    stego = encoder(cover, payload)

                # Validate WITHOUT distortion (clean decode accuracy)
                if mode == "hybrid":
                    predicted_logits, confidence = decoder(stego)
                else:
                    predicted_logits = decoder(stego)
                    confidence = None

                losses = criterion(
                    predicted_logits, payload, stego, cover, mask, confidence
                )
                val_losses.append(losses["total"].item())
                val_accs.append(losses["bit_accuracy"].item())

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = sum(val_accs) / len(val_accs)
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Log
        history["train_loss"].append(train_loss)
        history["train_bit_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_bit_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_acc": val_acc,
                "config": config,
            }
            ckpt_path = output_path / "checkpoints" / f"ckpt_epoch{epoch+1:03d}_acc{val_acc:.4f}.pt"
            torch.save(ckpt, ckpt_path)

        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = {
                "epoch": epoch + 1,
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "config": config,
                "val_acc": val_acc,
            }
            torch.save(best_ckpt, output_path / "best_model.pt")

    # Save final history
    with open(output_path / "logs" / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Outputs saved to: {output_path}")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "config": config,
    }
