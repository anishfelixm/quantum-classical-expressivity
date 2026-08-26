"""
FREEZING INTEGRITY.

Every frozen-backbone result rests on one claim: with freeze_policy="all", the
backbone does not change during training. That claim was implemented but never
tested.

WHY BUFFERS MATTER MORE THAN PARAMETERS
----------------------------------------
Parameters are protected by requires_grad=False - the optimizer cannot touch
them. BatchNorm running statistics are NOT: `running_mean` and `running_var`
update on the FORWARD pass, and requires_grad has no bearing on that.

The only thing keeping them fixed is eval() mode on the frozen blocks, applied
by set_bn_eval(). And model.train() - called at the top of every epoch - resets
every submodule to training mode, so set_bn_eval() must be re-applied AFTER it,
every epoch, forever.

If that call is ever dropped or reordered, the feature distribution drifts under
the heads, every "identical static features" claim silently becomes false, and
no metric in the pipeline would look wrong.

THE NEGATIVE CONTROL IS NOT OPTIONAL
-------------------------------------
A test that passes because nothing can move is worthless. test_negative_control_
buffers_drift_without_bn_eval runs the same model WITHOUT set_bn_eval() and
asserts the buffers DO change. If that test ever fails, the positive tests prove
nothing and both must be re-examined.

These use small synthetic images rather than MedMNIST: freezing is a property of
the module, not of the data, and a test that needs a download is a test that
gets skipped.

Run:  python -m pytest tests/test_freezing.py -v -s
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.backbone import TruncatedResNet18, LAYER3_INDEX   # noqa: E402
from models.registry import build_arm                          # noqa: E402
from train.loop import module_grad_norms                       # noqa: E402

IMG = 64          # ResNet-18 accepts anything >= 32; small keeps the test fast
BATCH = 4
STEPS = 3


def snapshot(module):
    return (
        {n: p.detach().clone() for n, p in module.named_parameters()},
        {n: b.detach().clone() for n, b in module.named_buffers()},
    )


def changed(before, after):
    """(param names changed, buffer names changed). Bit-exact comparison."""
    (bp, bb), (ap, ab) = before, after
    p = [n for n, t in bp.items() if not torch.equal(t, ap[n])]
    b = [n for n, t in bb.items() if not torch.equal(t.float(), ab[n].float())]
    return p, b


def train_briefly(model, num_classes=2, skip_bn_eval=False, lr=1e-2):
    """
    A few real optimizer steps on synthetic data.

    Mirrors the loop's ordering exactly: model.train() then set_bn_eval(), every
    step. The ordering is the fragile part, so the test reproduces it rather
    than calling train_model and hoping.
    """
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(0)

    for _ in range(STEPS):
        model.train()
        if not skip_bn_eval:
            model.set_bn_eval()
        x = torch.randn(BATCH, 3, IMG, IMG, generator=g)
        y = torch.randint(0, num_classes, (BATCH,), generator=g)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    return model


# ------------------------------------------------------------------ PROOF A
@pytest.mark.parametrize("arm", ["linear", "matched_param_fullrank",
                                 "low_rank", "quantum_vqc"])
def test_frozen_backbone_is_bit_identical_after_training(arm):
    """
    freeze_policy='all': not one parameter and not one buffer may change.

    Bit-exact on purpose. A frozen module has no mechanism that would produce a
    small change rather than none, so "close enough" would only hide a bug.
    """
    torch.manual_seed(0)
    model = build_arm(arm, d=4, num_classes=2, seed=42,
                      freeze_policy="all", build_backbone=True)

    before = snapshot(model.backbone)
    train_briefly(model)
    p, b = changed(before, snapshot(model.backbone))

    print(f"\n  {arm:24s} params changed={len(p)} buffers changed={len(b)}")
    assert not p, f"{arm}: frozen backbone PARAMETERS changed: {p[:5]}"
    assert not b, (f"{arm}: frozen backbone BUFFERS changed: {b[:5]} - "
                   f"set_bn_eval() is not holding")


def test_no_backbone_parameter_requires_grad_when_frozen():
    """The structural precondition, checked separately so a failure is specific."""
    model = build_arm("quantum_vqc", d=4, num_classes=2, seed=42,
                      freeze_policy="all", build_backbone=True)
    leaks = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  backbone params requiring grad: {len(leaks)}")
    print(f"  total trainable in model: {n_train}")
    assert not leaks, f"frozen backbone has trainable parameters: {leaks[:5]}"


def test_frozen_backbone_receives_no_gradient():
    """
    Complements the weight check: nothing changed AND nothing was even asked to.

    module_grad_norms returns None for a component with no gradient-carrying
    parameter, which is distinct from 0.0 ("received gradient that vanished").
    """
    model = build_arm("quantum_vqc", d=4, num_classes=2, seed=42,
                      freeze_policy="all", build_backbone=True)
    model.train()
    model.set_bn_eval()
    x = torch.randn(BATCH, 3, IMG, IMG)
    y = torch.randint(0, 2, (BATCH,))
    nn.functional.cross_entropy(model(x), y).backward()

    norms = module_grad_norms(model)
    print(f"\n  grad norms: {norms}")
    assert norms["backbone"] is None, "frozen backbone received gradient"
    assert norms["head"] is not None or norms["classifier"] is not None, (
        "nothing downstream received gradient either - the test is not "
        "exercising a real backward pass")


# ------------------------------------------------------- NEGATIVE CONTROL
def test_negative_control_buffers_drift_without_bn_eval():
    """
    THE TEST THAT MAKES THE OTHERS MEAN SOMETHING.

    Without set_bn_eval(), BatchNorm running statistics must drift - they update
    on the forward pass regardless of requires_grad. If this ever fails, then
    buffers cannot move under any circumstance, the positive tests above pass
    trivially, and none of them is evidence of anything.
    """
    torch.manual_seed(0)
    model = build_arm("linear", d=4, num_classes=2, seed=42,
                      freeze_policy="all", build_backbone=True)

    before = snapshot(model.backbone)
    train_briefly(model, skip_bn_eval=True)
    p, b = changed(before, snapshot(model.backbone))

    print(f"\n  without set_bn_eval: params changed={len(p)} "
          f"buffers changed={len(b)}")
    assert b, ("BatchNorm buffers did NOT drift without set_bn_eval() - the "
               "freezing tests are vacuous and prove nothing")
    assert not p, ("parameters changed even without set_bn_eval(); "
                   "requires_grad=False should still have held")


# ------------------------------------------------------------------ PROOF B
@pytest.mark.parametrize("arm", ["linear", "quantum_vqc"])
def test_adaptive_backbone_moves_and_receives_gradient(arm):
    """
    The complement of Proof A: with freeze_policy='layer3_only', layer3 MUST
    receive gradient and MUST move - including when the head is quantum.

    Everything outside layer3 must still be untouched, which is what makes
    'layer3_only' a meaningful label rather than a comment.
    """
    torch.manual_seed(0)
    model = build_arm(arm, d=4, num_classes=2, seed=42,
                      freeze_policy="layer3_only", build_backbone=True)

    model.train()
    model.set_bn_eval()
    x = torch.randn(BATCH, 3, IMG, IMG)
    y = torch.randint(0, 2, (BATCH,))
    nn.functional.cross_entropy(model(x), y).backward()
    norms = module_grad_norms(model)

    before = snapshot(model.backbone)
    train_briefly(model)
    p, _ = changed(before, snapshot(model.backbone))

    l3 = [n for n in p if n.startswith(f"features.{LAYER3_INDEX}")]
    other = [n for n in p if not n.startswith(f"features.{LAYER3_INDEX}")]

    print(f"\n  {arm:14s} backbone grad={norms['backbone']}")
    print(f"  layer3 params changed={len(l3)}  other blocks changed={len(other)}")
    assert norms["backbone"] is not None and norms["backbone"] > 0, (
        f"{arm}: no gradient reached the backbone - the head is not training "
        f"the encoder")
    assert l3, f"{arm}: layer3 did not move despite receiving gradient"
    assert not other, (f"{arm}: blocks outside layer3 changed: {other[:5]} - "
                       f"'layer3_only' is not holding")


def test_quantum_head_passes_gradient_to_its_input():
    """
    The specific step a sceptical reader will not take on trust: d(loss)/dz must
    survive the simulated circuit, or the VQC cannot train anything upstream of
    itself no matter what the backbone's requires_grad flags say.
    """
    model = build_arm("quantum_vqc", d=4, num_classes=2, seed=42,
                      build_backbone=False)
    h = torch.randn(BATCH, 256, requires_grad=True)
    y = torch.randint(0, 2, (BATCH,))
    nn.functional.cross_entropy(model.forward_from_features(h), y).backward()

    print(f"\n  d(loss)/dh norm = {h.grad.norm():.4e}")
    assert h.grad is not None, "no gradient reached the head's input at all"
    assert float(h.grad.abs().max()) > 0, (
        "gradient through the quantum head is identically zero")


if __name__ == "__main__":
    test_frozen_backbone_is_bit_identical_after_training("quantum_vqc")
    test_negative_control_buffers_drift_without_bn_eval()
    test_adaptive_backbone_moves_and_receives_gradient("quantum_vqc")
    test_quantum_head_passes_gradient_to_its_input()
    print("\nFreezing and flow verified.")
