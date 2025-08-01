#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


kmeans_feature="wavlm_large/23"  # use model_type/layer_index

#for nclusters in 100 200 500; do
for nclusters in 100; do
    src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
    tgt_lang=en

    train_set="train_960"
    train_dev="dev"
    test_sets="test_clean test_other dev_clean dev_other"

    asr_config=conf/tuning/train_discrete_asr_e_branchformer1.yaml
    inference_config=conf/decode_ctc0.3.yaml

    src_nbpe=$((nclusters * 5))   # src_nbpe is set to nclusters*5
    tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

    # ts: true sequence
    # rm: deduplicated sequence which removes duplicated tokens
    src_case="rm"
    tgt_case="ts"

    if [ "${nclusters}" -eq 100 ]; then
        skip_stages="2"
    else
        skip_stages="1 2"
    fi

    ./asr2.sh \
        --stage 6 \
        --kmeans_opts "--batch_bins 4800000 --skip_stages ${skip_stages}" \
        --kmeans_feature "${kmeans_feature}" \
        --nclusters "${nclusters}" \
        --ngpu 2 \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --src_token_type "bpe" \
        --src_nbpe $src_nbpe \
        --tgt_token_type "bpe" \
        --tgt_nbpe $tgt_nbpe \
        --src_case ${src_case} \
        --tgt_case ${tgt_case} \
        --speed_perturb_factors "0.9 1.0 1.1" \
        --asr_config "${asr_config}" \
        --inference_config "${inference_config}" \
        --train_set "${train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${test_sets}" \
        --src_bpe_train_text "dump/raw/${train_set}_sp/text.${src_case}.${src_lang}" \
        --tgt_bpe_train_text "dump/raw/${train_set}_sp/text.${tgt_case}.${tgt_lang}" \
        --lm_train_text "data/${train_set}/text" \
        --inference_asr_model "valid.acc.ave_10best.pth" \
        --gpu_inference true \
        --storage_save_mode false \
        "$@"
done

#    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
#    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
#    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang} data/local/other_text/text" \
