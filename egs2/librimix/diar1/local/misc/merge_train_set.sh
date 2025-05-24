#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

function merge_dir() {
    local dir="data/${1}"
    local sdir1="data/${2}"
    local sdir2="data/${3}"
    for file in reco2dur rttm segments spk2utt utt2spk wav.scp; do
        mkdir -p ${dir}
        echo -n "" > ${dir}/${file}
        cat ${sdir1}/${file} >> ${dir}/${file}
        cat ${sdir2}/${file} >> ${dir}/${file}
    done
}

dir=train-460
sdir1=train-100
sdir2=train-360
merge_dir $dir $sdir1 $sdir2
utils/fix_data_dir.sh data/$dir

dir=train-4602
sdir1=train-1002
sdir2=train-3602
merge_dir $dir $sdir1 $sdir2
utils/fix_data_dir.sh data/$dir

dir=train-4603
sdir1=train-1003
sdir2=train-3603
merge_dir $dir $sdir1 $sdir2
utils/fix_data_dir.sh data/$dir