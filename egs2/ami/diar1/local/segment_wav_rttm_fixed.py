import argparse
import logging
import os
import soundfile
import yaml
from tqdm import tqdm

logger = logging.getLogger("segment_wav_rttm_fixed.py")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("{asctime} ({name}:{lineno}) {message}", style="{")
handler.setFormatter(formatter)
logger.addHandler(handler)

def encode_segment_id(segment_id: int) -> str:
    return str(segment_id).zfill(3)

def segment_wav_rttm_fixed(
    config_path: str,
    mic_type: str,
    if_mini: str,
    sound_type: str,
    dataset_type: str,
    segment_output_dir: str,
    duration: float = 20.0,
    hop_size: float = 10.0,
) -> None:
    logger.info(f"Start segmenting {dataset_type} dataset with fixed duration and hop size")

    if if_mini == "true":
        if_mini = True
    elif if_mini == "false":
        if_mini = False
    else:
        raise ValueError("if_mini must be 'true' or 'false'")

    if sound_type == "word_and_vocalsounds":
        assert mic_type == "ihm", (
            "Only data with ihm microphone type is available for "
            "word_and_vocalsounds sound type."
        )
        assert not if_mini, (
            "Only full dataset is available for " "word_and_vocalsounds sound type."
        )
    if sound_type == "only_words":
        assert not if_mini, (
            "Only full dataset is available for "
            "only_words sound type, please set sound_type to None."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = "AMI-SDM" if mic_type == "sdm" else "AMI"
    task = "SpeakerDiarization"
    task_option = "mini" if if_mini else sound_type

    databases = config["Databases"]
    protocols = config["Protocols"]

    wav_id_txt = protocols[dataset][task][task_option][dataset_type]["uri"]
    wav_path_template = databases[dataset]
    rttm_path_template = protocols[dataset][task][task_option][dataset_type]["annotation"]

    output_segment_wavs_dir = os.path.join(segment_output_dir, dataset_type, "wav")
    output_segment_rttms_dir = os.path.join(segment_output_dir, dataset_type, "rttm")

    os.makedirs(output_segment_wavs_dir, exist_ok=True)
    os.makedirs(output_segment_rttms_dir, exist_ok=True)

    wav_ids = []
    segmented_wav_ids = []

    with open(wav_id_txt, "r") as f:
        for line in f:
            wav_ids.append(line.strip())

    for wav_id in tqdm(wav_ids, desc=f"{dataset_type} dataset"):
        wav_path = wav_path_template.format(uri=wav_id)
        rttm_path = rttm_path_template.format(uri=wav_id)

        wav, sr = soundfile.read(wav_path)
        assert len(wav.shape) == 1, f"Error in {wav_path}, the wav file should be mono channel"

        segment_id = 0
        total_duration = len(wav) / sr
        start_time = 0.0

        while start_time < total_duration:
            end_time = min(start_time + duration, total_duration)
            

            segment_rttm_entries = []
            with open(rttm_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    sps = line.strip().split()
                    assert len(sps) == 10, f"Error in {rttm_path}"

                    label_type, wav_id, channel, spk_start_time, spk_duration, _, _, spk_id, _, _ = sps
                    assert label_type == "SPEAKER", f"Error in {rttm_path}"

                    spk_start_time = float(spk_start_time)
                    spk_end_time = spk_start_time + float(spk_duration)

                    if spk_end_time > start_time and spk_start_time < end_time:
                        #relative_spk_start_time = round(spk_start_time - start_time, 2)
                        record_start_time = max(start_time, spk_start_time)
                        record_end_time = min(end_time, spk_end_time)
                        record_duration = round(record_end_time - record_start_time, 2)
                        relative_record_start_time = round(record_start_time - start_time, 2)
                        assert record_duration >= 0, f"Error in {rttm_path}, record_duration should be non-negative"
                        relative_line = f"{label_type} {wav_id}_{encode_segment_id(segment_id)} {channel} {relative_record_start_time} {record_duration} <NA> <NA> {spk_id} <NA> <NA>\n"
                        segment_rttm_entries.append(relative_line)

            if segment_rttm_entries:
                segment_rttm_path = os.path.join(output_segment_rttms_dir, f"{wav_id}_{encode_segment_id(segment_id)}.rttm")
                with open(segment_rttm_path, "w") as f:
                    f.writelines(segment_rttm_entries)
                segment_wav_path = os.path.join(output_segment_wavs_dir, f"{wav_id}_{encode_segment_id(segment_id)}.wav")
                soundfile.write(segment_wav_path, wav[int(start_time * sr):int(end_time * sr)], sr)
                segmented_wav_ids.append(f"{wav_id}_{encode_segment_id(segment_id)}")
            
            segment_id += 1
            start_time += hop_size

    segmented_wav_ids.sort()
    with open(os.path.join(segment_output_dir, dataset_type, "wav_ids.txt"), "w") as f:
        f.write("\n".join(segmented_wav_ids) + "\n")

    logger.info(f"Complete segmenting {dataset_type} dataset, {len(segmented_wav_ids)} segments are generated in total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami_diarization_config", type=str, required=True, help="Path to the config file of AMI diarization.")
    parser.add_argument("--mic_type", type=str, required=True, help="Microphone type, options: 'ihm', 'sdm'.")
    parser.add_argument("--if_mini", type=str, required=True, help="If true, use the subset of corresponding dataset.")
    parser.add_argument("--sound_type", type=str, required=True, help="Sound type, options: 'only_words', 'word_and_vocal'.")
    parser.add_argument("--segment_output_dir", type=str, required=True, help="Directory to store the output segmented wavs and rttms.")
    parser.add_argument("--duration", type=float, required=True, help="Duration of each segment.")
    parser.add_argument("--hop_size", type=float, required=True, help="Hop size for segmenting.")

    args = parser.parse_args()

    segment_wav_rttm_fixed(
        args.ami_diarization_config,
        args.mic_type,
        args.if_mini,
        args.sound_type,
        "train",
        args.segment_output_dir,
        args.duration,
        args.hop_size,
    )
    segment_wav_rttm_fixed(
        args.ami_diarization_config,
        args.mic_type,
        args.if_mini,
        args.sound_type,
        "dev",
        args.segment_output_dir,
        args.duration,
        args.hop_size,
    )
    segment_wav_rttm_fixed(
        args.ami_diarization_config,
        args.mic_type,
        args.if_mini,
        args.sound_type,
        "test",
        args.segment_output_dir,
        args.duration,
        args.hop_size,
    )