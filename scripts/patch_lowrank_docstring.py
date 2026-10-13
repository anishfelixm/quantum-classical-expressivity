"""
One-time edit of a single paragraph in src/models/heads.py (LowRankHead's docstring).

    python scripts/patch_lowrank_docstring.py

Why a script rather than a replacement file: heads.py is load-bearing (parameter
parity), and this change touches comments only. The script replaces exactly one
paragraph, refuses to run if that paragraph is not found exactly once, and
verifies afterwards that the file still compiles and that nothing outside the
paragraph changed.

The old text called restriction "the paper's central claim". It was tested
(H-S5, and the powered rerun of Amendment 17) and neither established nor
excluded, so the docstring now says that.
"""
import os
import py_compile
import re
import sys

PATH = os.path.join("src", "models", "heads.py")

OLD = re.compile(
    r"(?P<indent>[ \t]*)The paper's central claim is that the quantum head's advantage under"
    r".*?rather than anything quantum\.\s*?(?P<nl>\r?\n)\s*?\r?\n"
    r"[ \t]*That claim had no dedicated control\.",
    re.DOTALL)

NEW_LINES = [
    "The hypothesis this head was built to test (H-S5) was that the quantum",
    "head's advantage under extreme scarcity is a REGULARIZATION effect of its",
    "restricted function class - 24 parameters reaching a 24-dimensional",
    "manifold inside an 81-dimensional trigonometric span - rather than anything",
    "quantum. Tested twice (H-S5, and the powered rerun of Amendment 17), it was",
    "neither established nor excluded: no scarcity-dependent effect of",
    "restriction was found.",
    "",
    "Before this head existed, that hypothesis had no dedicated control.",
]


def main():
    with open(PATH, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    if NEW_LINES[0] in text:
        print(f"{PATH}: already updated; nothing to do.")
        return
    hits = list(OLD.finditer(text))
    if len(hits) != 1:
        sys.exit(f"expected the paragraph exactly once in {PATH}, found {len(hits)}; "
                 f"nothing changed. Send the current heads.py.")
    m = hits[0]
    ind, nl = m.group("indent"), m.group("nl")
    new = nl.join((ind + line) if line else "" for line in NEW_LINES)
    out = text[:m.start()] + new + text[m.end():]
    # Only the matched span may differ.
    assert out[:m.start()] == text[:m.start()] and out[m.start() + len(new):] == text[m.end():]
    with open(PATH, "wb") as f:
        f.write(out.encode("utf-8"))
    py_compile.compile(PATH, doraise=True)
    print(f"{PATH}: one docstring paragraph replaced ({m.end() - m.start()} -> {len(new)} "
          f"characters); file compiles.")


if __name__ == "__main__":
    main()
