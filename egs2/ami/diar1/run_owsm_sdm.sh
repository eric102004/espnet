#!/usr/bin/env bash

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail


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
num_spk=None

# Arguments for /local/data.sh
setup_dir=ami_diarization_setup
mic_type=sdm
if_mini=false
sound_type=only_words
fixed=true
duration=30
hop_size=20
min_wav_duration=0.0 # set to 0.0 to use all data, don't filter out short utterances.

lr=000003
train_config="conf/${duration}_${hop_size}/v7_train_diar_eda_lr${lr}.yaml"
#train_config="conf/train_diar_eda2.yaml"
decode_config="conf/decode_diar.yaml"

train_set="train_${mic_type}_${sound_type}_${num_spk}_${duration}"
valid_set="dev_${mic_type}_${sound_type}_${num_spk}_${duration}"
test_sets="test_${mic_type}_${sound_type}_${num_spk}_${duration}"

# Arguments for scoring
scoring_frame_shift=640 # frame shift for scoring, which is same as the frame shift in the model.
#scoring_frame_shift=128 # frame shift for scoring, which is same as the frame shift in the model.
scoring_subsampling=1 # subsampling factor for scoring, which is same as the subsampling factor in the model.
sampling_frequency=16k

dumpdir=dump_${mic_type}_${sound_type}_${num_spk}_${duration}
expdir=exp_eda_${mic_type}_${sound_type}_${num_spk}_${duration}
diar_tag=diar_v7_train_diar_eda_lr${lr}_librimix_pretrained_2gpu
#diar_tag=diar_train_diar_eda2

if [ ${fixed} == true ]; then
    train_set="${train_set}_${hop_size}"
    valid_set="${valid_set}_${hop_size}"
    test_sets="${test_sets}_${hop_size}"
    dumpdir="${dumpdir}_${hop_size}"
    expdir="${expdir}_${hop_size}"
fi

./diar.sh \
    --collar 0.0 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --ngpu 2 \
    --diar_config "${train_config}" \
    --inference_config "${decode_config}" \
    --inference_nj 5 \
    --local_data_opts "
        --stage 3 \
        --setup_dir ${setup_dir}
        --num_spk ${num_spk}
        --mic_type ${mic_type}
        --if_mini ${if_mini}
        --sound_type ${sound_type}
        --duration ${duration}
        --train_set ${train_set}
        --valid_set ${valid_set}
        --test_sets ${test_sets}
        --hop_size ${hop_size}
        --fixed ${fixed}
    " \
    --num_spk 5 \
    --dumpdir "${dumpdir}" \
    --expdir "${expdir}" \
    --min_wav_duration "${min_wav_duration}"\
    --frame_shift "${scoring_frame_shift}"\
    --subsampling "${scoring_subsampling}"\
    --fs "${sampling_frequency}"\
    --diar_tag "${diar_tag}" \
    --hop_length 160 \
    "$@"

# original:
#    ngpu=3 