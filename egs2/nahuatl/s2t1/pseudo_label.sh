#!/usr/bin/env bash
# Decode unlabeled (untranscribed) Nahuatl recordings per region with the
# fine-tuned OWSM S2T model to produce pseudo-label hypotheses.
#
# Per-region: prep_unlabeled.py VAD-segments the recordings not already used
# for train/val/test (Task 1), then s2t_inference.py decodes each segment.
# s2t_inference's DatadirWriter (espnet2/fileio/datadir_writer.py) writes one
# line per key "uttid value" per file under "{n}best_recog/"; with the default
# --nbest 1 that's "1best_recog/{text,token,token_int,score,text_nospecial}".
# All three files we need (text, score, token_int) are keyed by uttid and, once
# each is independently line-sorted, stay aligned since "sort" orders by the
# leading uttid field the same way in each file.
#
# prep_unlabeled.py's wav.scp entries are Kaldi unix-pipe commands
# ("UTTID ffmpeg ... -f wav - |"), which ESPnet2's SoundScpReader (the
# ",sound" data type used below) cannot read directly - it calls
# soundfile.SoundFile() on the scp value and does not support pipes. Mirror
# stage 3 of egs2/TEMPLATE/s2t1/s2t.sh: materialize each region's wav.scp to
# real per-utterance .wav files with scripts/audio/format_wav_scp.sh BEFORE
# decode_dir, then decode against the materialized wav.scp.
#SBATCH -N 1 -n 1 -p gpuA40x4,gpuA100x4
#SBATCH --gres=gpu:4 -c 32 --mem 120000M
#SBATCH --account=bbjs-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=nahuatl-pl
#SBATCH --output=%x_%j.log
set -o pipefail
RECIPE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$RECIPE_DIR"; source path.sh
# path.sh does not set cuda_cmd (only cmd.sh does) - decode_dir needs it below.
source cmd.sh
export PATH="/work/hdd/bbjs/clin10/kaldi/tools/sctk/bin:$PATH"

RAW=/work/nvme/bbjs/shared/nahuatl/nahuatl
SPLITS=/work/nvme/bbjs/clin10/nahuatl_asr/splits.json
WORK="${WORK:-/work/hdd/bbjs/clin10/pl}"      # intermediates off /work/nvme
PSEUDO_DIR="${PSEUDO_DIR:-data/pseudo}"       # combined pseudo Kaldi dir output
S2T_EXP=$(ls -d exp/s2t_train_owsm_v4_nahuatl_raw_bpe50000_init_param* | head -1)
MODEL="$S2T_EXP/valid.acc.ave.pth"
MAX_HOURS="${MAX_HOURS:-30}"                  # subset cap (per region)

# region -> "slug:lang_sym:decode_cfg_suffix". lang_sym here is informational
# only (it documents which <nah_*> tag conf/decode_owsm_${cfg}.yaml bakes in
# via its `lang_sym:` key); decode_dir does not consume it directly.
declare -A REG=( [Hidalgo]="hidalgo:<nah_hid>:hid"
                 [Orizaba-Zongolica]="orizaba_zongolica:<nah_ozg>:ozg"
                 [Zacatlan-Tepetzintla]="zacatlan_tepetzintla:<nah_ztp>:ztp" )

# One decode process per allocated GPU, each pinned to a distinct device.
# We do NOT use run.pl here: it sizes concurrency from `nvidia-smi -L` (which
# sees all node GPUs, ignoring the cgroup) and never pins a per-job
# CUDA_VISIBLE_DEVICES, so with nj>1 every job piled onto one GPU -> OOM/thrash.
# Instead split the wav.scp into (#GPU) parts and launch one background
# s2t_inference per GPU with CUDA_VISIBLE_DEVICES set to that GPU's id, then
# wait on all and fail if any split failed. batch_size stays 1 (s2t_inference
# raises NotImplementedError for batch decoding).
# Use CUDA_VISIBLE_DEVICES (cgroup-relative ids, 0..N-1 under SLURM) as the GPU
# list, not SLURM_JOB_GPUS (physical ids the cgroup may have remapped).
IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NJ=${#GPU_IDS[@]}

decode_dir() {  # $1=wav.scp  $2=out  $3=decode_cfg
  local scp="$1" out="$2" cfg="$3"
  mkdir -p "$out/logdir"
  utils/split_scp.pl "$scp" \
    $(for j in $(seq "$NJ"); do echo "$out/logdir/wav.$j.scp"; done)
  local pids=() j
  for j in $(seq "$NJ"); do
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$((j-1))]}" \
      python -m espnet2.bin.s2t_inference --ngpu 1 --batch_size 1 \
        --data_path_and_name_and_type "$out/logdir/wav.$j.scp,speech,sound" \
        --key_file "$out/logdir/wav.$j.scp" \
        --s2t_train_config "$S2T_EXP/config.yaml" --s2t_model_file "$MODEL" \
        --config "$cfg" --output_dir "$out/logdir/out.$j" \
        > "$out/logdir/infer.$j.log" 2>&1 &
    pids+=("$!")
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  if [ "$rc" -ne 0 ]; then
    echo "decode_dir: a split failed; see $out/logdir/infer.*.log" >&2
    return 1
  fi
  for f in text score token_int; do
    for j in $(seq "$NJ"); do cat "$out/logdir/out.$j/1best_recog/$f"; done \
      | sort > "$out/$f"
  done
}

for R in "${!REG[@]}"; do
  IFS=: read slug tok cfg <<< "${REG[$R]}"
  udir="$WORK/unlabeled_${slug}"
  python local/prep_unlabeled.py --raw_region_dir "$RAW/$R" --splits_file "$SPLITS" \
      --output_dir "$udir" --max_hours "$MAX_HOURS"
  utils/validate_data_dir.sh --no-feats --no-text "$udir"
  # Materialize the pipe-style wav.scp to real 16k mono .wav files (see note
  # above) - writes "$fmtdir/wav.scp" with real file paths. train_cmd comes
  # from cmd.sh (sourced above); it's "run.pl" for this recipe's local backend.
  fmtdir="$WORK/fmt_${slug}"
  scripts/audio/format_wav_scp.sh --nj 8 --cmd "${train_cmd}" \
      --audio-format wav --fs 16k "$udir/wav.scp" "$fmtdir"
  decode_dir "$fmtdir/wav.scp" "$WORK/decode_${slug}" "conf/decode_owsm_${cfg}.yaml"
  echo "decoded $slug: $(wc -l < "$WORK/decode_${slug}/text") segments"
done

# --- tune threshold on val (decode val once) ---
# dump/raw/nahuatl_valid/wav.scp already has real paths (materialized by the
# baseline training's stage 3), so decode_dir can read it directly - no
# format_wav_scp.sh needed here.
decode_dir dump/raw/nahuatl_valid/wav.scp "$WORK/decode_valid" \
    conf/decode_owsm_hid.yaml
THR=$(python local/tune_threshold.py --decode_dir "$WORK/decode_valid" \
      --ref_text dump/raw/nahuatl_valid/text --target_cer "${TARGET_CER:-0.15}" \
      | awk '/^THRESHOLD/{print $2}')
echo "chosen threshold: $THR"
if [ -z "$THR" ] || [ "$THR" = "inf" ]; then
    echo "ERROR: no confidence threshold met the target CER (THR='$THR'); no pseudo-labels would pass. Aborting." >&2
    exit 1
fi

# --- filter each region + combine into data/pseudo ---
pdirs=()
for R in "${!REG[@]}"; do
  IFS=: read slug tok cfg <<< "${REG[$R]}"
  python local/filter_pseudo.py --decode_dir "$WORK/decode_${slug}" \
     --unlabeled_dir "$WORK/unlabeled_${slug}" --region_token "$tok" \
     --threshold "$THR" --output_dir "$WORK/pseudo_${slug}"
  pdirs+=("$WORK/pseudo_${slug}")
done
utils/combine_data.sh "$PSEUDO_DIR" "${pdirs[@]}"
utils/validate_data_dir.sh --no-feats "$PSEUDO_DIR"
echo "pseudo utterances: $(wc -l < "$PSEUDO_DIR/text")"
