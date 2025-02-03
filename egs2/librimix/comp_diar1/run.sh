#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

config_name=train_asr_lr002
asr_config=conf/tuning/${config_name}.yaml
inference_config=conf/decode_asr.yaml
dumpdir=dump_rle2_rep30
asr_tag=${config_name}_${dumpdir}
inference_asr_model="valid.cer.ave_10best.pth"

max_wav_duration=10000     # there should be max duration limit for diarization

fs=16k

# asr training (please run diar1/run.sh --stop_stage 4 and local/gen_spk_label.sh to generate text before running this)
# start from stagee 5 to generate token_list
./asr.sh \
    --stage 5 \
    --fs $fs \
    --ngpu 1 \
    --token_type "word" \
    --use_lm false \
    --max_wav_duration ${max_wav_duration} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --dumpdir "${dumpdir}" \
    --asr_tag "${asr_tag}" \
    --lm_train_text "${dumpdir}/raw/train/text" \
    "$@"

    #--lm_config "${lm_config}" \
    #--bpe_train_text "data/${train_set}/text_spk1 data/${train_set}/text_spk2" \
    #--collar 0.0 \
