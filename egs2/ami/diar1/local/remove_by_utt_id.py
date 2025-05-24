import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Remove data with specific utterance IDs from multiple files.")
parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing the files to update")
args = parser.parse_args()

# Script to remove data with specific utterance IDs from multiple files

# Define the utterance IDs to remove
remove_utt_ids = {'IS1007d', 'IS1003b'}

# Define the base directory
base_dir = args.base_dir
files_to_update = ["segments", "utt2spk", "wav.scp"]

def filter_file(file_path, remove_ids):
    """Filter lines in a file based on utterance IDs."""
    temp_file = file_path + ".tmp"
    with open(file_path, "r") as infile, open(temp_file, "w") as outfile:
        for line in infile:
            utt_id = line.split()[0]
            if os.path.basename(file_path) == "wav.scp":
                utt_id = utt_id.split("_")[0]
            elif os.path.basename(file_path) in ["segments", "utt2spk"]:
                utt_id = utt_id.split("_")[1]
            if utt_id not in remove_ids:
                outfile.write(line)
    os.replace(temp_file, file_path)

def main():
    for file_name in files_to_update:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            filter_file(file_path, remove_utt_ids)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()