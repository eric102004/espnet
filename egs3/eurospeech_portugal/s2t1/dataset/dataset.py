import numpy as np
from datasets import Audio, load_from_disk
from torch.utils.data import Dataset

from espnet2.bin.s2t_inference import Speech2Text

class EuroSpeechPortugalDataset(Dataset):
    def __init__(self, data_dir, split, ratio=1.0):
        if not (0 < ratio <= 1.0):
            raise ValueError("ratio must be in the range (0, 1].")

        dataset_dict = load_from_disk(data_dir)
        if split in dataset_dict:
            self.dataset = dataset_dict[split]
        else:
            raise ValueError(f"Split '{split}' not found in dataset.")

        self.dataset = self.dataset.cast_column("audio", Audio())
        if ratio < 1.0:
            keep = int(len(self.dataset) * ratio)
            self.dataset = self.dataset.select(range(keep))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        transcript = item["human_transcript"]
        example = {
            "speech": item["audio"]["array"].astype(np.float32),
            "text": f"<por><asr><notimestamps> {transcript}",
            "text_ctc": transcript,
            "text_prev": "<na>",
        }
        return example


class OWSMTokenizeTransform:
    def __init__(self, model_tag, *args, **kwargs):
        owsm_model = Speech2Text.from_pretrained(model_tag)
        self.tokenizer = owsm_model.tokenizer
        self.converter = owsm_model.converter

    def tokenize(self, text):
        return np.array(self.converter.tokens2ids(self.tokenizer.text2tokens(text)))

    def __call__(self, data):
        example = data
        ret = dict(
            speech=example['speech'],
            text=self.tokenize(example['text']),
            text_ctc=self.tokenize(example['text_ctc']),
            text_prev=self.tokenize(example['text_prev']),
        )
        return ret

class WhisperTokenizeTransform:
    def __init__(self, model_tag, *args, **kwargs):
        from transformers import WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(model_tag)

    def tokenize(self, text):
        tokens = self.processor.tokenizer.encode(text, return_tensors="np")
        return tokens.flatten()

    def __call__(self, data):
        example = data

        # preprocessing and calculate new speech_lengths
        # pad to 30 seconds (3000 frames after processing)
        # devide speech_lengths by 160, and build attention_mask
        # convert speech to list of numpy arrays for processor
        # speech = [s.detach().cpu().numpy() for s in speech]  # list of (T,)
        processed = self.processor(
            example['speech'],
            sampling_rate=16000,
            return_tensors="np",
            padding=True,
        )
        speech = np.transpose(np.squeeze(processed.input_features, axis=0), (1, 0))  # (B, D, T') --> (T', D)
        # pad to 30 seconds (3000 frames after processing)
        # speech = torch.nn.functional.pad(speech, (0, max(0, 3000 - speech.size(2))), value=0.0)[:, :, :3000]  # (B, D, 3000)
        # speech_lengths = torch.tensor([min(l // 160, 3000) for l in speech_lengths])  # (B,)

        ret = dict(
            speech=speech,
            text=self.tokenize(example['text']),
            text_ctc=self.tokenize(example['text_ctc']),
            text_prev=self.tokenize(example['text_prev']),
        )
        return ret