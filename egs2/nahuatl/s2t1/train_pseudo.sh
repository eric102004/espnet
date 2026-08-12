#!/usr/bin/env bash
# Combine labeled train + pseudo-labels, retrain OWSM from the patched
# checkpoint on the combined set, and evaluate on the 3 region test sets.
#
# Pipeline:
#   1) combine data/nahuatl_train (labeled) + data/pseudo (Task 4 output) into
#      data/nahuatl_train_plus_pseudo via combine_data.sh, then regenerate
#      text.prev (<na> for all) and text.ctc (region/task-token-stripped
#      transcript) exactly as local/data.sh does — s2t.sh stages 10/11 need
#      both utt_extra_files streams.
#   2) validate the combined dir.
#   3) retrain from the patched OWSM checkpoint (run.sh already sets
#      --init_param to it) on the combined set via the train_set override
#      added in run.sh, stages 3-11 (re-format wav.scp, collect stats, train;
#      feats_stats symlink + region-token bpe are unchanged from the baseline
#      run and are reused as-is).
#   4) decode + score the 3 region test sets plus the combined aggregate via
#      decode.sh (reuses local/score.sh's symmetric CER).
#SBATCH -N 1 -n 1 -p gpuA40x4,gpuA100x4
#SBATCH --gres=gpu:1 -c 16 --mem 60000M
#SBATCH --account=bbjs-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=nahuatl-pl-train
#SBATCH --output=%x_%j.log
set -o pipefail

# SLURM copies the batch script to a spool dir, so ${BASH_SOURCE[0]} is unreliable
# under sbatch. Use SLURM_SUBMIT_DIR (the dir sbatch was launched from) when set,
# and fall back to the script's own location for plain `bash train_pseudo.sh`.
RECIPE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$RECIPE_DIR"; source path.sh
# sclite (used by local/score.sh, invoked from decode.sh) lives under sctk;
# path.sh does not put it on PATH.
export PATH="/work/hdd/bbjs/clin10/kaldi/tools/sctk/bin:$PATH"

# ── 1) Combined train = labeled + pseudo; regenerate text.prev/text.ctc ────
utils/combine_data.sh data/nahuatl_train_plus_pseudo data/nahuatl_train data/pseudo
python3 - <<'PY'
import re
d = "data/nahuatl_train_plus_pseudo"
tok = re.compile(r'<(nah_hid|nah_ozg|nah_ztp|asr|notimestamps)>\s*')
with open(f"{d}/text") as f, open(f"{d}/text.prev", "w") as fp, \
     open(f"{d}/text.ctc", "w") as fc:
    for line in f:
        uid, *rest = line.strip().split(None, 1)
        clean = tok.sub('', rest[0] if rest else '').strip()
        fp.write(f"{uid} <na>\n")
        fc.write(f"{uid} {clean}\n")
PY
utils/validate_data_dir.sh --no-feats data/nahuatl_train_plus_pseudo

# ── 2) Retrain from patched OWSM on the combined set ────────────────────────
# run.sh's --init_param already points at the patched OWSM checkpoint (not the
# current fine-tuned model), so overriding only train_set here gives a fresh
# retrain from OWSM on labeled+pseudo data.
export train_set=nahuatl_train_plus_pseudo
bash run.sh --stage 3 --stop_stage 11 2>&1 | tee pseudo_train_live.log

# ── 3) Evaluate on the 3 region test sets + combined aggregate ─────────────
bash decode.sh
