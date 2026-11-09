"""
AMENDMENT 31: a circuit arm on raw z, built only on request.

What must hold:
1. By default a circuit arm still refuses use_tanh=False. Every run before
   Amendment 31 was built under that rule, and a re-execution must find it.
2. With allow_unbounded_angles=True the circuit builds, keeps its 24 head
   parameters at d = 4 (parity is unchanged: the squash has no parameters), and
   its latent is raw z, not (pi/2) tanh(z).
3. The permission changes nothing for a classical arm or for an arm with the
   squash: the latent is the same function either way.

Run:  python -m pytest tests/test_unbounded_angles.py -v
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.registry import build_arm, count_head_params   # noqa: E402


def test_circuit_refuses_raw_z_by_default():
    with pytest.raises(ValueError, match="allow_unbounded_angles"):
        build_arm("quantum_vqc", d=4, num_classes=2, seed=42, build_backbone=False, use_tanh=False)


def test_circuit_on_raw_z_with_permission():
    m = build_arm("quantum_vqc", d=4, num_classes=2, seed=42, build_backbone=False,
                  use_tanh=False, allow_unbounded_angles=True)
    assert count_head_params(m)["head"] == 24
    assert m.use_tanh is False
    h = torch.randn(5, 256)
    z = m.bottleneck(h)
    assert torch.equal(m.latent(h), z)


def test_permission_changes_nothing_else():
    h = torch.randn(5, 256)
    for arm in ("matched_param_fullrank", "quantum_vqc"):
        b = build_arm(arm, d=4, num_classes=2, seed=42, build_backbone=False,
                      allow_unbounded_angles=True)
        assert b.use_tanh is True
        assert torch.equal(b.latent(h), torch.tanh(b.bottleneck(h)) * b.angle_scale)
        assert abs(b.angle_scale - math.pi / 2) < 1e-12
    m = build_arm("matched_param_fullrank", d=4, num_classes=2, seed=42, build_backbone=False,
                  use_tanh=False)
    assert count_head_params(m)["head"] == 24 and m.use_tanh is False
