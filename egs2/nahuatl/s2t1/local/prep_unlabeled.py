#!/usr/bin/env python3
"""Segment untranscribed Nahuatl recordings into a Kaldi dir for pseudo-labeling.

Drops any recording whose primary consultant (first filename code) is in the
val/test splits, runs silero-VAD to cut recordings into <=30 s speech chunks,
and writes an ffmpeg-pipe wav.scp (16 kHz mono) plus utt2spk / spk2utt. No text.
"""
import argparse
import collections
import glob
import io
import json
import os
import re
import subprocess

_CODE = re.compile(r"^([A-Za-z]{2,4}\d{3,})$")


def primary_code(fname):
    for part in os.path.splitext(os.path.basename(fname))[0].split("_"):
        for s in part.split("-"):
            if _CODE.match(s):
                return s
    return None


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9]", "_", s)


def load_16k_mono(path):
    import soundfile as sf

    cmd = ["ffmpeg", "-nostdin", "-loglevel", "quiet", "-i", path,
           "-ar", "16000", "-ac", "1", "-f", "wav", "-"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    data, sr = sf.read(io.BytesIO(out), dtype="float32")
    return data, sr


def vad_segments(wav, sr, get_ts, model, max_dur=30.0, min_dur=1.0):
    import torch

    ts = get_ts(torch.from_numpy(wav), model, sampling_rate=sr)
    segs = []
    for t in ts:
        s, e = t["start"] / sr, t["end"] / sr
        while e - s > max_dur:
            segs.append((s, s + max_dur))
            s += max_dur
        if e - s >= min_dur:
            segs.append((s, e))
    return segs


def val_test_consultants(splits_file):
    splits = json.load(open(splits_file))
    excl = set()
    for path, sp in splits.items():
        if sp.endswith("-val") or sp.endswith("-test"):
            c = primary_code(path)
            if c:
                excl.add(c)
    return excl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_region_dir", required=True)
    p.add_argument("--splits_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_hours", type=float, default=None)
    p.add_argument("--max_recordings", type=int, default=None)
    args = p.parse_args()

    exclude = val_test_consultants(args.splits_file)
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    os.makedirs(args.output_dir, exist_ok=True)

    wavs = sorted(
        glob.glob(os.path.join(args.raw_region_dir, "Sounds", "**", "*.wav"),
                  recursive=True)
    )
    rows = []
    total = 0.0
    nrec = 0
    for w in wavs:
        pc = primary_code(w)
        if pc is None or pc in exclude:
            continue
        if args.max_recordings and nrec >= args.max_recordings:
            break
        wav, sr = load_16k_mono(w)
        base = sanitize(os.path.splitext(os.path.basename(w))[0])
        for s, e in vad_segments(wav, sr, get_speech_timestamps, model):
            utt = f"{pc}_{base}_{int(s * 100):07d}"
            rows.append((utt, pc, os.path.abspath(w), s, e))
            total += e - s
        nrec += 1
        if args.max_hours and total / 3600 >= args.max_hours:
            break

    rows.sort(key=lambda r: r[0])
    with open(f"{args.output_dir}/wav.scp", "w") as fw, \
         open(f"{args.output_dir}/utt2spk", "w") as fu:
        for utt, spk, wav_path, s, e in rows:
            fw.write(
                f"{utt} ffmpeg -nostdin -loglevel quiet -ss {s:.3f} "
                f"-t {e - s:.3f} -i {wav_path} -ar 16000 -ac 1 -f wav - |\n"
            )
            fu.write(f"{utt} {spk}\n")
    s2u = collections.defaultdict(list)
    for utt, spk, _, _, _ in rows:
        s2u[spk].append(utt)
    with open(f"{args.output_dir}/spk2utt", "w") as f:
        for spk, us in sorted(s2u.items()):
            f.write(f"{spk} {' '.join(sorted(us))}\n")
    print(f"Wrote {len(rows)} segments ({total / 3600:.1f} h), {nrec} recordings")


if __name__ == "__main__":
    main()
