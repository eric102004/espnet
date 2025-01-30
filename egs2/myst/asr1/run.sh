#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set="train"
valid_set="dev"
test_sets="dev test"

encoder=transformer
frontend=wavlm
asr_config=conf/tuning/train_asr_${frontend}_${encoder}.yaml
inference_config=conf/decode_asr.yaml
asr_tag="train_asr_${frontend}_${encoder}_raw_en_char_dur05_filter"
inference_asr_model="valid.cer.ave_4best.pth"
#asr_config=conf/tuning/train_asr.yaml
#inference_config=conf/decode_asr.yaml
#asr_tag="train_asr_raw_en_char_dur05_filter"
#inference_asr_model="valid.cer.ave_10best.pth"

dumpdir=dump_filter

#nbpe=5000
#bpemode=unigram

# if your sox supports flac file, set local_data_opts and audio_format as below.
#local_data_opts=""
#audio_format=flac

# if your sox does not support flac file, set local_data_opts and audio_format as below.
local_data_opts="--flac2wav true"
audio_format=wav

# set a higher min_wav_duration to avoid TooShortUttError in stage 11
min_wav_duration=0.5

./asr.sh \
    --lang en \
    --gpu_inference true \
    --token_type char \
    --max_wav_duration 30 \
    --use_lm false \
    --feats_normalize utt_mvn \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --local_data_opts "${local_data_opts}" \
    --audio_format ${audio_format} \
    --min_wav_duration ${min_wav_duration} \
    --asr_tag "${asr_tag}" \
    --dumpdir "${dumpdir}" \
    "$@"

#    --speed_perturb_factors "0.9 1.0 1.1" \
#    --bpemode "${bpemode}" \
#    --nbpe "${nbpe}" \