import pytest
import torch
from training.losses import ThreeTierLoss


def test_same_sub_uses_ncc():
    loss_fn = ThreeTierLoss()
    warped = torch.rand(1, 1, 16, 16, 16)
    fixed = torch.rand(1, 1, 16, 16, 16)
    pair_types = torch.tensor([0])
    result = loss_fn(warped, fixed, pair_types)
    assert result.shape == ()
    assert torch.isfinite(result)


def test_cross_sub_uses_mi():
    loss_fn = ThreeTierLoss()
    warped = torch.rand(1, 1, 16, 16, 16)
    fixed = torch.rand(1, 1, 16, 16, 16)
    pair_types = torch.tensor([1])
    result = loss_fn(warped, fixed, pair_types)
    assert result.shape == ()
    assert torch.isfinite(result)


def test_cross_super_uses_mi():
    loss_fn = ThreeTierLoss()
    warped = torch.rand(1, 1, 16, 16, 16)
    fixed = torch.rand(1, 1, 16, 16, 16)
    pair_types = torch.tensor([2])
    result = loss_fn(warped, fixed, pair_types)
    assert result.shape == ()
    assert torch.isfinite(result)


def test_mixed_batch():
    loss_fn = ThreeTierLoss()
    warped = torch.rand(3, 1, 16, 16, 16)
    fixed = torch.rand(3, 1, 16, 16, 16)
    pair_types = torch.tensor([0, 1, 2])
    result = loss_fn(warped, fixed, pair_types)
    assert result.shape == ()
    assert torch.isfinite(result)


def test_ema_tracking():
    loss_fn = ThreeTierLoss(ema_decay=0.9)
    warped = torch.rand(2, 1, 16, 16, 16)
    fixed = torch.rand(2, 1, 16, 16, 16)
    for _ in range(5):
        pair_types = torch.tensor([0, 1])
        loss_fn(warped, fixed, pair_types)
    assert loss_fn.ema_ncc > 0
    assert loss_fn.ema_mi > 0
