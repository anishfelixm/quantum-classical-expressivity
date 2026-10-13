"""
v50 text fixes: comments, docstrings and two historical docs. No code path changes.

    python scripts/v50_text_fixes.py

WHAT IT CHANGES, AND WHY
------------------------
src/07_hardware_noise.py   docstring only.
    - The channel was written rho -> (1-p) rho + p I/2, which contracts <X> by
      (1-p). PennyLane's DepolarizingChannel is the Pauli form, which contracts
      by (1 - 4p/3), the factor the docstring itself quotes. Wording now
      matches the code.
    - Adds the scope of the AUC-invariance argument: exact only when the channel
      acts once after the final block (true for the single-block VQC). With
      per-layer noise an independent simulation reorders 24 of 1,770 pairs
      (tests/test_independent_theory.py).
    - Removes the claim that this is the conference "Zombie State" mechanism.
      That observation was under sensor noise, where the journal's controls find
      the F1 collapse in every small head (docs/RESULTS.md section 3).
src/09_lr_selection.py     docstring only: drops "the VQC's calibration failure",
                           which the ECE data do not support.
src/config.py              comment only: the same phrase.
docs/HYPOTHESES.md         a banner saying the file is historical (12 August)
                           and superseded; the body is kept as written.
docs/MATH_VERIFICATION.md  a banner and a dated status addendum (section 11).
                           The body is kept as written. The addendum corrects
                           two statements: section 1 ("no complex phase is ever
                           used") and section 7 ("monotone decay in both n and
                           L"). It also records which open items were resolved.

SAFETY
------
Each edit is an exact, single-occurrence replacement. If the expected text is
not found exactly once, that edit is skipped and reported, and nothing else in
that file changes. Line endings (CRLF or LF) are preserved. Python files are
compiled afterwards. Running it twice is harmless: applied edits are detected
and skipped. heads.py and registry.py are not touched, so the parameter-parity
check is not required; it is cheap, so run it anyway if you like.
"""
import os
import py_compile
import sys


def find(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


ADDENDUM_MATH = """## 11. Status update — 13 October 2026

Added after all experiments finished. Sections 1–10 above are kept as first
written, so the record shows what was checked and when.

**Two statements above are wrong as written. Neither changes a result.**

- **§1, "no complex phase is ever used".** The variational layers
  (`Rot = RZ·RY·RZ` plus CNOTs) do introduce complex phases. What is true is
  narrower: the ENCODED state is real. For a real vector ψ and Hermitian
  M = A + iB (A real symmetric, B real antisymmetric), ψᵀBψ = 0. So only the
  real symmetric part of M(Θ) reaches the output. §2 is unaffected. The
  manuscript says "the encoded amplitudes are real" and nothing stronger.
- **§7, "monotone decay in both n and L".** It holds in n at every L, and in L
  at n = 8 and 16. At n = 4 the variance rises with L (10.45 → 11.21 → 12.96).
  The ~62× figure is unaffected. The script and log that produced the table
  have to be identified before the paper cites it (`docs/WORK_REMAINING.md`).

**Open items above, now resolved:**

- §4, re-uploading: run as H-S1 (it helps at n ≥ 10, the opposite of its
  prediction) and as E7 against a function-class control. See `docs/RESULTS.md`.
- §5, angle scale: π vs π/2 was swept. π helps the VQC in 3 of 20 cells and
  hurts the control in 5 of 20, so π/2 did not handicap the VQC.
- §6, ±ω duplicates: found and fixed. One representative per ±ω pair
  (`canonical_frequencies`). H-S2 was run on the corrected implementation.
- §8: the confirmatory analysis uses the nested bootstrap (`13_family_table.py`).
  The normal-approximation intervals are not quoted.

**Added since.**

- **Hardware depolarising noise (`07`).** It contracts every ⟨Xᵢ⟩ by
  (1 − 4p/3) and leaves binary AUC exactly invariant only when the channel acts
  once, after the final block. That is the model simulated for the
  single-block VQC. With noise after every layer, an independent simulation
  gives input-dependent factors (0.81–0.87 at p = 0.05) and reorders 24 of
  1,770 test pairs.
- **Independent re-derivation.** `tests/test_independent_theory.py` rebuilds the
  circuit in plain numpy, with no PennyLane or project code. It gives the 3^d
  span to 2×10⁻¹⁵, the two-qubit readout to 2×10⁻¹⁵, the re-upload 5^d span to
  4×10⁻¹⁵ (the 3^d span fails at 0.27), and the end-of-circuit contraction to
  2×10⁻¹⁶.
"""

BANNER_MATH = """> **Status, 13 October 2026.** Written before the confirmatory run. Two
> statements below are corrected and the open items resolved in §11 at the end.
"""

BANNER_HYP = """> **Historical — drafted August 2026, statuses as of 12 August.** Superseded by
> `docs/analysis_plan.md` (the binding plan) and `docs/RESULTS.md` (outcomes).
> Statuses, run counts and "prior signal" readings below are not current. In
> particular, the "calibration failure" reading of the conference noise result
> is not supported (`docs/RESULTS.md` §3). H0 is reported there as the
> compression premise check.
"""

EDITS = [
    # (label, candidate paths, kind, old/anchor, new/addition, marker)
    ("07 channel formula", ("src/07_hardware_noise.py",), "replace",
     "Single-qubit depolarizing sends rho -> (1-p) rho + p I/2. Since tr(X I/2) = 0,\n"
     "EVERY measured expectation is contracted by the SAME positive factor:",
     "PennyLane's DepolarizingChannel is the Pauli form\n"
     "rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z). Applied once to every\n"
     "qubit after the final block, it contracts EVERY measured expectation by the\n"
     "SAME positive factor:",
     "PennyLane's DepolarizingChannel is the Pauli form"),
    ("07 scope note", ("src/07_hardware_noise.py",), "insert_after",
     "multi-class the softmax mixes logits non-linearly so AUC moves, but barely\n"
     "(BloodMNIST 0.8382 -> 0.8385 at p=0.05).",
     "\n\nSCOPE. The invariance needs the channel to act once, after the final block.\n"
     "That is true here for the single-block VQC (n_uploads = 1). With noise after\n"
     "every layer, or after each block of a re-uploading circuit, the contraction\n"
     "depends on the input and AUC can move: an independent simulation with\n"
     "per-layer noise reorders 24 of 1,770 test pairs at p = 0.05\n"
     "(tests/test_independent_theory.py).",
     "SCOPE. The invariance needs the channel to act once"),
    ("07 Zombie State paragraph", ("src/07_hardware_noise.py",), "replace",
     "It is also the \"Zombie State\" mechanism of the conference paper, derived from\n"
     "first principles rather than observed: a model that still ranks correctly while\n"
     "its probabilities have collapsed.",
     "The conference paper described a similar pattern under SENSOR noise (its\n"
     "\"Zombie State\"). That observation does not carry over: under sensor noise the\n"
     "journal's controls find the F1 collapse in every small head, with the VQC's\n"
     "calibration error among the lowest (docs/RESULTS.md section 3). The mechanism\n"
     "above belongs to this hardware noise model only.",
     "The conference paper described a similar pattern under SENSOR noise"),
    ("09 criterion wording", ("src/09_lr_selection.py",), "replace",
     "CRITERION. Validation AUC, never test. AUC because it is the primary endpoint\n"
     "and because selecting on F1 interacts with the VQC's calibration failure.",
     "CRITERION. Validation AUC, never test. AUC because it is the primary endpoint,\n"
     "and because F1 also depends on where the decision threshold falls - an extra\n"
     "source of noise on 10-20 validation images. (An earlier version of this\n"
     "sentence cited a \"calibration failure\" of the VQC; the ECE data do not\n"
     "support that - docs/RESULTS.md section 3.)",
     "F1 also depends on where the decision threshold falls"),
    ("config selection comment", ("src/config.py", "config.py"), "replace",
     "# reporting AUC let the selection criterion interact with the VQC's calibration\n"
     "# failure - a confound, not a preference.",
     "# reporting AUC selected checkpoints on a different metric from the one\n"
     "# reported - a confound, not a preference.",
     "# reporting AUC selected checkpoints on a different metric from the one"),
    ("HYPOTHESES banner", ("docs/HYPOTHESES.md", "HYPOTHESES.md"), "insert_after",
     "# Hypotheses\n", "\n" + BANNER_HYP,
     "**Historical — drafted August 2026"),
    ("MATH_VERIFICATION banner", ("docs/MATH_VERIFICATION.md", "MATH_VERIFICATION.md"),
     "insert_after", "# Mathematical Verification\n", "\n" + BANNER_MATH,
     "**Status, 13 October 2026.**"),
    ("MATH_VERIFICATION addendum", ("docs/MATH_VERIFICATION.md", "MATH_VERIFICATION.md"),
     "append", None, ADDENDUM_MATH, "## 11. Status update — 13 October 2026"),
]


def main():
    report, failed, touched = [], 0, set()
    for label, cands, kind, old, new, marker in EDITS:
        path = find(*cands)
        if path is None:
            report.append(f"MISSING  {label}: none of {cands} exists")
            failed += 1
            continue
        raw = open(path, "rb").read().decode("utf-8")
        nl = "\r\n" if "\r\n" in raw else "\n"
        conv = (lambda s: s.replace("\n", nl)) if nl == "\r\n" else (lambda s: s)
        if conv(marker) in raw:
            report.append(f"SKIP     {label}: already applied ({path})")
            continue
        if kind == "append":
            out = raw.rstrip("\r\n") + nl + nl + conv(new).rstrip("\r\n") + nl
        else:
            n = raw.count(conv(old))
            if n != 1:
                report.append(f"NOT DONE {label}: expected text found {n} times in "
                              f"{path}; file left as it was for this edit")
                failed += 1
                continue
            rep = conv(new) if kind == "replace" else conv(old) + conv(new)
            out = raw.replace(conv(old), rep)
        with open(path, "wb") as f:
            f.write(out.encode("utf-8"))
        touched.add(path)
        report.append(f"DONE     {label} ({path})")

    for path in sorted(touched):
        if path.endswith(".py"):
            py_compile.compile(path, doraise=True)
            report.append(f"COMPILES {path}")

    print("\n".join(report))
    if failed:
        print(f"\n{failed} edit(s) not applied. Send the file(s) named above.")
        sys.exit(1)
    print("\nAll edits applied or already present.")


if __name__ == "__main__":
    main()
