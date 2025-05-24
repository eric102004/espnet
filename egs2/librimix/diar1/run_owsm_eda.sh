#!/usr/bin/env bash

# Copyright 2021 Yushi Ueda
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

train_set="train-460"
valid_set="dev"
test_sets="test test2 test3"

lr=00003
warmup_steps=8000
train_config="conf/tuning/owsm/v7_train_diar_eda_lr${lr}_warmup${warmup_steps}.yaml"
decode_config="conf/decode_diar_eda.yaml"

# Arguments for scoring
scoring_frame_shift=640 # frame shift for scoring, which is same as the frame shift in the model.

dumpdir=dump_libri23mix_460
expdir=exp_libri23mix_460


./diar.sh \
    --fs 16k \
    --collar 0.0 \
    --gpu_inference true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 5 \
    --dumpdir "${dumpdir}" \
    --expdir "${expdir}" \
    --frame_shift "${scoring_frame_shift}"\
    --num_spk 3 \
    --hop_length 160 \
    "$@"
