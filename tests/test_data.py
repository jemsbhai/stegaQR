"""Tests for synthetic dataset and data loading."""

import pytest
import torch
from stegaqr.data.synthetic import SyntheticStegaQRDataset, create_dataloaders


class TestSyntheticDataset:
    def test_dataset_length(self):
        ds = SyntheticStegaQRDataset(num_samples=10, capacity_bits=100, qr_version=4, module_size=1)
        assert len(ds) == 10

    def test_sample_keys(self):
        ds = SyntheticStegaQRDataset(num_samples=5, capacity_bits=100, qr_version=4, module_size=1)
        sample = ds[0]
        assert "cover" in sample
        assert "payload" in sample
        assert "mask" in sample
        assert "public_text" in sample

    def test_cover_shape_module1(self):
        ds = SyntheticStegaQRDataset(num_samples=5, capacity_bits=100, qr_version=4, module_size=1)
        sample = ds[0]
        d = 33  # 4*4+17
        assert sample["cover"].shape == (3, d, d)

    def test_cover_shape_module4(self):
        ds = SyntheticStegaQRDataset(num_samples=5, capacity_bits=100, qr_version=4, module_size=4)
        sample = ds[0]
        d = 33 * 4  # 132
        assert sample["cover"].shape == (3, d, d)

    def test_cover_shape_module8(self):
        ds = SyntheticStegaQRDataset(num_samples=5, capacity_bits=100, qr_version=4, module_size=8)
        sample = ds[0]
        d = 33 * 8  # 264
        assert sample["cover"].shape == (3, d, d)

    def test_cover_range(self):
        ds = SyntheticStegaQRDataset(num_samples=5, capacity_bits=100, qr_version=4, module_size=1)
        sample = ds[0]
        assert sample["cover"].min() >= 0.0
        assert sample["cover"].max() <= 1.0

    def test_payload_shape(self):
        ds = SyntheticStegaQRDataset(num_samples=5, capacity_bits=64, qr_version=2, module_size=1)
        sample = ds[0]
        assert sample["payload"].shape == (64,)

    def test_payload_binary(self):
        ds = SyntheticStegaQRDataset(num_samples=5, capacity_bits=100, qr_version=4, module_size=1)
        sample = ds[0]
        unique = torch.unique(sample["payload"])
        assert all(v in [0.0, 1.0] for v in unique)

    def test_mask_shape_module1(self):
        ds = SyntheticStegaQRDataset(num_samples=5, capacity_bits=100, qr_version=4, module_size=1)
        sample = ds[0]
        assert sample["mask"].shape == (1, 33, 33)

    def test_mask_shape_module4(self):
        ds = SyntheticStegaQRDataset(num_samples=5, capacity_bits=100, qr_version=4, module_size=4)
        sample = ds[0]
        assert sample["mask"].shape == (1, 132, 132)

    @pytest.mark.parametrize("version", [2, 4, 6])
    def test_spatial_size_property(self, version: int):
        ms = 4
        ds = SyntheticStegaQRDataset(num_samples=1, capacity_bits=50, qr_version=version, module_size=ms)
        assert ds.spatial_size == (4 * version + 17) * ms


class TestDataLoaders:
    def test_create_dataloaders(self):
        train_loader, val_loader = create_dataloaders(
            num_train=64, num_val=16, capacity_bits=100,
            qr_version=4, module_size=1, batch_size=8,
        )
        batch = next(iter(train_loader))
        assert batch["cover"].shape == (8, 3, 33, 33)
        assert batch["payload"].shape == (8, 100)
        assert batch["mask"].shape == (8, 1, 33, 33)
        assert len(batch["public_text"]) == 8

    def test_create_dataloaders_scaled(self):
        train_loader, val_loader = create_dataloaders(
            num_train=16, num_val=8, capacity_bits=100,
            qr_version=4, module_size=4, batch_size=4,
        )
        batch = next(iter(train_loader))
        assert batch["cover"].shape == (4, 3, 132, 132)
