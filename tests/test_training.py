"""Smoke test for the training pipeline.

Runs a minimal training loop (2 epochs, tiny dataset) to verify
the full pipeline works end-to-end without errors.
"""

import pytest
from pathlib import Path
from stegaqr.training import train


@pytest.mark.slow
class TestTrainingSmoke:

    def _run_smoke(self, mode: str, tmp_path: Path):
        capacity = 99 if mode == "segregated" else 100
        result = train(
            mode=mode,
            capacity_bits=capacity,
            qr_version=2,
            module_size=2,  # small for speed
            epochs=2,
            batch_size=4,
            num_train=16,
            num_val=8,
            seed=42,
            device="cpu",
            output_dir=str(tmp_path / f"smoke_{mode}"),
            checkpoint_every=2,
            use_distortion=False,
            warmup_decode_only=1,  # 1 epoch warmup, 1 epoch full
        )
        assert result["best_val_acc"] >= 0.0
        assert len(result["history"]["train_loss"]) == 2
        assert (tmp_path / f"smoke_{mode}" / "best_model.pt").exists()
        assert (tmp_path / f"smoke_{mode}" / "config.json").exists()
        assert (tmp_path / f"smoke_{mode}" / "seed.json").exists()
        return result

    def test_segregated_smoke(self, tmp_path):
        self._run_smoke("segregated", tmp_path)

    def test_cross_channel_smoke(self, tmp_path):
        self._run_smoke("cross_channel", tmp_path)

    def test_hybrid_smoke(self, tmp_path):
        self._run_smoke("hybrid", tmp_path)

    def test_with_distortion(self, tmp_path):
        result = train(
            mode="cross_channel",
            capacity_bits=50,
            qr_version=2,
            module_size=2,
            epochs=2,
            batch_size=4,
            num_train=16,
            num_val=8,
            seed=42,
            device="cpu",
            output_dir=str(tmp_path / "smoke_distortion"),
            checkpoint_every=2,
            use_distortion=True,
            warmup_decode_only=1,
        )
        assert result["best_val_acc"] >= 0.0
