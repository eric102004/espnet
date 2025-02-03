from argparse import ArgumentParser
from espnet2.train.class_choices import ClassChoices

from espnet2.diar.compressor.abs_compressor import AbsCompressor
from espnet2.diar.compressor.binary_to_decimal_compressor import BinaryToDecimalCompressor
from espnet2.diar.compressor.bpe_compressor import BPECompressor
from espnet2.diar.compressor.rle_compressor import RLECompressor, RLECompressor2

parser = ArgumentParser()
parser.add_argument("--dumpdir", type=str)
parser.add_argument("--special_token", type=str, default="s")
parser.add_argument("--compressor", type=str, default="none")
parser.add_argument("--blank_id", type=int, default=-1)
parser.add_argument("--max_repeat", type=int, default=-1)
parser.add_argument("--model_file", type=str, default="")
parser.add_argument("--bpe_vocab_size", type=int, default=-1)
parser.add_argument("--compression_rate", type=int, default=-1)
args = parser.parse_args()

assert len(args.special_token) == 1
dsets = ["train", "dev", "test"]

# init compressor
if args.compressor == "none":
    raise ValueError("compressor must be specified")
elif args.compressor == "rle":
    assert args.blank_id != -1
    assert args.max_repeat != -1
    compressor = RLECompressor(
        blank_id = args.blank_id,
        max_repeat = args.max_repeat, 
    )
elif args.compressor == "rle2":
    assert args.blank_id != -1
    assert args.max_repeat != -1
    compressor = RLECompressor2(
        blank_id = args.blank_id,
        max_repeat = args.max_repeat,
    )
elif args.compressor == "bpe":
    assert args.model_file != ""
    assert args.bpe_vocab_size != -1
    compressor = BPECompressor(
        model_file = args.model_file,
        bpe_vocab_size = args.bpe_vocab_size,
    )
elif args.compressor == "binary_to_decimal":
    assert args.blank_id != -1
    assert args.compression_rate != -1
    compressor = BinaryToDecimalCompressor(
        blank_id = args.blank_id,
        compression_rate = args.compression_rate,
    )
else:
    raise ValueError(f"compressor {args.compressor} is not supported")


# start for loop
for dset in dsets:
    org_spk_label_file = f"{args.dumpdir}/raw/{dset}/org_spk_label"
    output_file = f"{args.dumpdir}/raw/{dset}/text"
    with open(org_spk_label_file, "r") as f, open(output_file, "w") as f_out:
        for l in f.readlines():
            # read spk_label
            utt_id, spk_label = l.strip().split()
            spk_label = spk_label.split(args.special_token)
            spk_label = [[int(s) for s in sp] for sp in spk_label]
            spk_label_length = [len(sp) for sp in spk_label]
            # compress spk_label with compressor
            comp_spk_label, comp_spk_label_length = compressor.encode(spk_label, spk_label_length)
            sep = " " + args.special_token + " "
            comp_spk_label = sep.join(' '.join(str(s) for s in sp) for sp in comp_spk_label)
            # write compressed spk_label to text file
            f_out.write(f"{utt_id} {comp_spk_label}\n")