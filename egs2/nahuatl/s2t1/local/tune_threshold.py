#!/usr/bin/env python3
"""Pick a length-normalized log-prob threshold on the labeled val set.

Reads decoded val hyps ({score,text,token_int}) + reference text, computes
per-utterance normalized score and char CER, and prints the highest threshold
whose kept subset has mean CER <= --target_cer."""
import argparse
import os
import re


def read_kv(path):
    d = {}
    for line in open(path):
        parts = line.rstrip("\n").split(" ", 1)
        d[parts[0]] = parts[1] if len(parts) > 1 else ""
    return d


def strip_special(t):
    return re.sub(r"<[^>]*>", "", t).strip()


def char_cer(ref, hyp):
    r = list(ref)
    h = list(hyp)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[len(h)] / max(1, len(r))


def norm_score(score, ntok):
    return float(score) / max(1, ntok)


def pick_threshold(rows, target_cer):
    """rows: list of (norm_score, cer). Return (threshold, kept_frac, est_cer)."""
    rows = sorted(rows, reverse=True)  # high score first
    s = 0.0
    best = None
    for i, (ns, c) in enumerate(rows):
        s += c
        mean = s / (i + 1)
        if mean <= target_cer:
            best = (ns, (i + 1) / len(rows), mean)
    return best if best else (float("inf"), 0.0, 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decode_dir", required=True)
    p.add_argument("--ref_text", required=True)
    p.add_argument("--target_cer", type=float, default=0.15)
    args = p.parse_args()
    score = read_kv(os.path.join(args.decode_dir, "score"))
    hyp = read_kv(os.path.join(args.decode_dir, "text"))
    tok = read_kv(os.path.join(args.decode_dir, "token_int"))
    ref = read_kv(args.ref_text)
    rows = []
    for u in hyp:
        if u not in score or u not in ref:
            continue
        ns = norm_score(score[u], len(tok.get(u, "").split()))
        rows.append((ns, char_cer(strip_special(ref[u]), strip_special(hyp[u]))))
    thr, frac, est = pick_threshold(rows, args.target_cer)
    print(f"keeps {frac * 100:.0f}% of val, est CER {est * 100:.1f}%")
    print(f"THRESHOLD {thr}")


if __name__ == "__main__":
    main()
