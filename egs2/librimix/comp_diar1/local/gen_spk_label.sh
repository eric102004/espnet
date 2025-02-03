

org_dumpdir=dump

target_dumpdir=dump_rle2_rep30
compressor="rle2"


# convert espnet_rttm to spk_label
python local/rttm_to_spk_label.py \
    --dumpdir ${org_dumpdir} \
    "$@"


# copy org_dumpdir
rm -r ${target_dumpdir}
cp -r ${org_dumpdir} ${target_dumpdir}

# generate compressed spk label and save to text
python local/compress_spk_label.py \
    --dumpdir ${target_dumpdir} \
    --compressor ${compressor} \
    --blank_id 60 \
    --max_repeat 30 \
    "$@"
