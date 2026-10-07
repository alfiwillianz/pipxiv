"""Sanity checks for TLDR normalization (single clean paragraph).

Run inside the container:  docker run --rm -v "$PWD:/app" -w /app pipxiv-pipxiv python3 tests/test_tldr_normalize.py
"""

from bot.modules.arxiv import _normalize_tldr


CASES = {
    "fenced_block": ("```\nA summary inside a code fence.\n```", "A summary inside a code fence."),
    "inline_code": ("Uses `self-rollout` repacking.", "Uses self-rollout repacking."),
    "bare_tldr_line": ("TLDR\nA sentence here.", "A sentence here."),
    "tldr_colon": ("TLDR: A short single sentence summary.", "A short single sentence summary."),
    "bulleted": ("- Point one.\n- Point two.", "Point one. Point two."),
    "numbered": ("1. First.\n2. Second.", "First. Second."),
    "headings": ("## Findings\nRAVEN helps.", "Findings RAVEN helps."),
    "clean": ("A clean one sentence tldr.", "A clean one sentence tldr."),
    "two_sentences_kept": (
        "RAVEN repacks rollouts. It also adds a GRPO variant.",
        "RAVEN repacks rollouts. It also adds a GRPO variant.",
    ),
    "trailing_meta_cut": (
        "A short tldr sentence. If you'd like, I can help more.",
        "A short tldr sentence.",
    ),
    "empty": ("", None),
    "whitespace_only": ("   \n  ", None),
}


def main() -> int:
    failures = 0
    for name, (raw, expected) in CASES.items():
        got = _normalize_tldr(raw)
        ok = got == expected
        if not ok:
            failures += 1
        print(f"{'PASS' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            print(f"     expected: {expected!r}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
