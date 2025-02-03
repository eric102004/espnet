from argparse import ArgumentParser
import torch

from espnet2.fileio.rttm import RttmReader
from espnet2.layers.label_aggregation import LabelAggregate

parser = ArgumentParser()
parser.add_argument("--dumpdir", type=str)
parser.add_argument("--special_token", type=str, default="s")
parser.add_argument("--win_length", type=int, default=512)
parser.add_argument("--hop_length", type=int, default=128)
args = parser.parse_args()

assert len(args.special_token) == 1
dsets = ["train", "dev", "test"]

# init label aggregator
aggregator = LabelAggregate(
    win_length = args.win_length,
    hop_length = args.hop_length,
)

for dset in dsets:
    # read wav.scp file to get current order of utt_id
    with open(f"{args.dumpdir}/raw/{dset}/wav.scp", "r") as f:
        utt_ids = [line.split()[0] for line in f.readlines()]
    # read rttm and generate spk_label
    rttm_reader = RttmReader(fname=f"{args.dumpdir}/raw/{dset}/espnet_rttm")
    # generate spk_label
    org_spk_label_file = f"{args.dumpdir}/raw/{dset}/org_spk_label"
    with open(org_spk_label_file, "w") as f:
        for utt_id in utt_ids:
            spk_label = torch.from_numpy(rttm_reader[utt_id].astype(int)).unsqueeze(0)      # (1, T, num_spk)
            spk_label_length = torch.tensor([spk_label.size(1)])

            # convert spk_label to specified sample rate
            spk_label, spk_label_length = aggregator(spk_label, spk_label_length)
            spk_label = spk_label.squeeze(0).permute(-1, -2).tolist()      # (1, T, num_spk) -> (num_spk, T)

            # sort spk_label following first-in-first-out
            spk_label = sorted(spk_label, key=lambda x: x.index(1) if 1 in x else len(x))

            # concat spk_labels of multiple speaker with special token
            spk_label = [''.join(str(int(s)) for s in sp) for sp in spk_label]
            spk_label = args.special_token.join(spk_label)

            # write to org_spk_label file
            f.write(f"{utt_id} {spk_label}\n")
            
