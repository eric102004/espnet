#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"


# setting asr config
learning_rate=0001
asr_config=conf/tuning/owsm_ctc_lr${learning_rate}.yaml

# setting inference config
beam_size=1
inference_config=conf/decode_ctc_beam${beam_size}.yaml
inference_s2t_model=valid.cer_ctc.ave_4best.pth

bpemodel=data/en_token_list/bpe_unigram50000/owsm_ctc/bpe.model
bpetoken_list=data/en_token_list/bpe_unigram50000/owsm_ctc/tokens.txt

dumpdir=dump_filter

nbpe=50000

# if your sox supports flac file, set local_data_opts and audio_format as below.
#local_data_opts=""
#audio_format=flac

# if your sox does not support flac file, set local_data_opts and audio_format as below.
local_data_opts="--flac2wav true"
audio_format=wav

# set a higher min_wav_duration to avoid TooShortUttError in stage 11
min_wav_duration=0.5

./s2t_ctc.sh \
    --lang en \
    --gpu_inference true \
    --token_type bpe \
    --nbpe ${nbpe} \
    --max_wav_duration 30 \
    --use_lm false \
    --feats_normalize utt_mvn \
    --feats_type raw \
    --s2t_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_s2t_model "${inference_s2t_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --local_data_opts "${local_data_opts}" \
    --audio_format ${audio_format} \
    --min_wav_duration ${min_wav_duration} \
    --dumpdir ${dumpdir} \
    --bpemodel ${bpemodel} \
    --bpetoken_list ${bpetoken_list} \
    "$@"

#    --bpe_nlsyms data/nlsyms.txt \
#    --s2t_args "--model_conf extract_feats_in_collect_stats=false --batch_size 5 ${wandb_init_args}" \