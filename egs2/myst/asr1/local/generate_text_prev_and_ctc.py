
import shutil

dumpdir = "dump_filter"

dsets = ["train", "dev", "test"]

for dset in dsets:
    # read utt id from text file
    with open(f"{dumpdir}/raw/{dset}/text") as f:
        utt_ids = [line.split()[0] for line in f]

    # write utt_id <tab> <na> to text_prev file
    with open(f"{dumpdir}/raw/{dset}/text.prev", "w") as f:
        for utt_id in utt_ids:
            f.write(f"{utt_id}\t<na>\n")

    # copy text to text_ctc file
    shutil.copyfile(f"{dumpdir}/raw/{dset}/text", f"{dumpdir}/raw/{dset}/text.ctc")