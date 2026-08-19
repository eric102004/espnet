#!/usr/bin/env python3
"""Filter pseudo-labels by a normalized log-prob threshold + sanity checks and
write a Kaldi data dir with region-prefixed text."""
import argparse
import collections
import os
import re
from collections import Counter


def read_kv(path):
    d = {}
    for line in open(path):
        parts = line.rstrip("\n").split(" ", 1)
        d[parts[0]] = parts[1] if len(parts) > 1 else ""
    return d


def parse_score(s):
    # s2t_inference writes str(hyp.score); hyp.score is a 0-dim torch tensor, so
    # the file value is like "tensor(-14.1931)" rather than a bare float.
    s = s.strip()
    if s.startswith("tensor(") and s.endswith(")"):
        s = s[len("tensor(") : -1]
    return float(s)


def is_degenerate(text):
    toks = text.split()
    if not toks:
        return True
    if Counter(toks).most_common(1)[0][1] > max(3, 0.5 * len(toks)):
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decode_dir", required=True)
    p.add_argument("--unlabeled_dir", required=True)
    p.add_argument("--region_token", required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--min_chars", type=int, default=3)
    p.add_argument("--max_chars", type=int, default=400)
    args = p.parse_args()
    score = read_kv(os.path.join(args.decode_dir, "score"))
    hyp = read_kv(os.path.join(args.decode_dir, "text"))
    tok = read_kv(os.path.join(args.decode_dir, "token_int"))
    wav = read_kv(os.path.join(args.unlabeled_dir, "wav.scp"))
    u2s = read_kv(os.path.join(args.unlabeled_dir, "utt2spk"))
    os.makedirs(args.output_dir, exist_ok=True)
    kept = []
    for u, h in hyp.items():
        h = re.sub(r"<[^>]*>", "", h).strip()
        if u not in score or u not in wav:
            continue
        ns = parse_score(score[u]) / max(1, len(tok.get(u, "").split()))
        if ns < args.threshold:
            continue
        if not (args.min_chars <= len(h) <= args.max_chars):
            continue
        if is_degenerate(h):
            continue
        kept.append((u, h))
    kept.sort()
    with open(f"{args.output_dir}/wav.scp", "w") as fw, \
         open(f"{args.output_dir}/text", "w") as ft, \
         open(f"{args.output_dir}/utt2spk", "w") as fu:
        for u, h in kept:
            fw.write(f"{u} {wav[u]}\n")
            ft.write(f"{u} {args.region_token}<asr><notimestamps> {h}\n")
            fu.write(f"{u} {u2s[u]}\n")
    s2u = collections.defaultdict(list)
    for u, _ in kept:
        s2u[u2s[u]].append(u)
    with open(f"{args.output_dir}/spk2utt", "w") as f:
        for spk, us in sorted(s2u.items()):
            f.write(f"{spk} {' '.join(sorted(us))}\n")
    print(f"Kept {len(kept)}/{len(hyp)} pseudo-labels -> {args.output_dir}")


if __name__ == "__main__":
    main()
