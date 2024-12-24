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

train_config="conf/train_diar.yaml"
decode_config="conf/decode_diar.yaml"
: '
num_spk:
    4:
        if specifies 4, will remove wav files that are not with 4 speakers in
        AMI (there are several wav files with 3 or 5 speakers in train and test set).
        For the details about the number of speakers in each wav files, please refer
        the data set statistics in /DEBUG.
    None:
        Using Encoder-Decoder Attractors to process unknown-speaker-number diarization.
'
num_spk=4

# Arguments for /local/data.sh
setup_dir=ami_diarization_setup
mic_type=ihm
if_mini=false
sound_type=only_words
duration=20
min_wav_duration=0.0 # set to 0.0 to use all data, don't filter out short utterances.

# Arguments for scoring
scoring_frame_shift=80 # frame shift for scoring, which is same as the frame shift in the model.
scoring_subsampling=10 # subsampling factor for scoring, which is same as the subsampling factor in the model.

./diar.sh \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 1 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 5 \
    --local_data_opts "
        --setup_dir ${setup_dir}
        --num_spk ${num_spk}
        --mic_type ${mic_type}
        --if_mini ${if_mini}
        --sound_type ${sound_type}
        --duration ${duration}
    " \
    --num_spk "${num_spk}"\
    --min_wav_duration "${min_wav_duration}"\
    --frame_shift "${scoring_frame_shift}"\
    --subsampling "${scoring_subsampling}"\
    "$@"
