#!/usr/bin/env bash

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

lr=00002
train_config_name=train_diar_ebranchformer_lr${lr}
train_config="conf/${train_config_name}.yaml"
#train_config="conf/train_diar_eda_baseline.yaml"
decode_config="conf/decode_diar.yaml"
diar_tag="${train_config_name}_2spk_16k"

num_spk=2 # 2, 3
fs=16k                        # switch to 16k
inference_model=valid.acc.ave_10best.pth


dumpdir=dump_${num_spk}spk
expdir=exp_${num_spk}spk

scoring_frame_shift=512 # since we are training ebranchformer with conv1d (downsample by 4)

./diar.sh \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 5 \
    --fs $fs \
    --local_data_opts "
        --num_spk ${num_spk}
        --train_set ${train_set}
        --valid_set ${valid_set}
        --test_sets ${test_sets}
    " \
    --num_spk "${num_spk}"\
    --diar_tag "${diar_tag}" \
    --frame_shift ${scoring_frame_shift} \
    --dumpdir "${dumpdir}" \
    --expdir "${expdir}" \
    "$@"
