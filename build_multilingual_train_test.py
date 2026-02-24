#!/usr/bin/env python3
import argparse
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")  # number + whitespace + text

def read_numbered_sentences(path: str, normalize: str = "NFC") -> List[str]:
    sentences: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue

            m = LINE_RE.match(raw)
            if m:
                txt = m.group(2)
            else:
                parts = raw.strip().split(None, 1)
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                txt = parts[1]

            txt = unicodedata.normalize(normalize, txt)
            if txt:
                sentences.append(txt)
    return sentences

def sample_pairs(
    sentences: List[str],
    n_pairs_per_sentence: int,
    rng: random.Random,
    min_prefix: int,
    max_prefix: int,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for s in sentences:
        if len(s) < (min_prefix + 1):
            continue
        hi = min(len(s) - 1, max_prefix)
        if hi < min_prefix:
            continue
        for _ in range(n_pairs_per_sentence):
            k = rng.randint(min_prefix, hi)
            pairs.append((s[:k], s[k]))
    return pairs

def lang_tag_from_filename(path: str) -> str:
    name = Path(path).name
    return name.split("_", 1)[0] if "_" in name else Path(name).stem

def write_inputs_answers(outdir: str, prefix: str, tag: str, pairs: List[Tuple[str, str]]) -> None:
    os.makedirs(outdir, exist_ok=True)
    inp = os.path.join(outdir, f"{prefix}_input_{tag}.txt")
    ans = os.path.join(outdir, f"{prefix}_answer_{tag}.txt")

    with open(inp, "w", encoding="utf-8") as fin, open(ans, "w", encoding="utf-8") as fans:
        for pfx, nxt in pairs:
            fin.write(pfx + "\n")
            fans.write(nxt + "\n")

    print(f"[{tag}] {prefix}: {len(pairs)} examples")
    print(f"  {inp}")
    print(f"  {ans}")

def split_train_test(pairs: List[Tuple[str, str]], test_frac: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    # pairs should already be shuffled
    n = len(pairs)
    n_test = int(round(n * test_frac))
    n_test = max(1, n_test) if n > 0 else 0
    test = pairs[:n_test]
    train = pairs[n_test:]
    return train, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Numbered sentence files (one per language).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs-per-sentence", type=int, default=1)
    ap.add_argument("--max-sentences", type=int, default=100000)
    ap.add_argument("--min-prefix", type=int, default=1)
    ap.add_argument("--max-prefix", type=int, default=80)
    ap.add_argument("--test-frac", type=float, default=0.05, help="Fraction of examples to put in test set (e.g. 0.05).")
    ap.add_argument("--normalize", default="NFC", choices=["NFC", "NFKC", "NFD", "NFKD"])
    args = ap.parse_args()

    base_rng = random.Random(args.seed)

    for path in args.inputs:
        tag = lang_tag_from_filename(path)

        sentences = read_numbered_sentences(path, normalize=args.normalize)
        if not sentences:
            print(f"[{tag}] WARNING: no usable sentences parsed from {path}")
            continue

        if len(sentences) > args.max_sentences:
            sentences = sentences[:args.max_sentences]

        # Deterministic-but-distinct RNG per language
        rng = random.Random(base_rng.randint(0, 2**31 - 1))

        pairs = sample_pairs(
            sentences,
            n_pairs_per_sentence=args.pairs_per_sentence,
            rng=rng,
            min_prefix=args.min_prefix,
            max_prefix=args.max_prefix,
        )
        rng.shuffle(pairs)

        train_pairs, test_pairs = split_train_test(pairs, test_frac=args.test_frac)

        # Write aligned train + test (input/answer only)
        write_inputs_answers(args.outdir, "train", tag, train_pairs)
        write_inputs_answers(args.outdir, "test", tag, test_pairs)

if __name__ == "__main__":
    main()