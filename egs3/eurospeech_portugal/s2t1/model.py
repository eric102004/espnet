import torch
import torch.nn as nn

def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)
    ).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)

class OWSMFinetune(nn.Module):
    def __init__(self, model_tag):
        super().__init__()
        from espnet2.bin.s2t_inference import Speech2Text
        owsm_model = Speech2Text.from_pretrained(model_tag)
        self.model = owsm_model.s2t_model

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.model.train()
        else:
            self.model.eval()

        return self

    def forward(
        self,
        speech,
        speech_lengths,
        text,
        text_lengths,
        text_ctc,
        text_ctc_lengths,
        text_prev,
        text_prev_lengths,
    ):
        return self.model(
            speech,
            speech_lengths,
            text,
            text_lengths,
            text_ctc,
            text_ctc_lengths,
            text_prev,
            text_prev_lengths,
        )

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}


class WhisperFinetune(nn.Module):
    def __init__(self, model_tag):
        super().__init__()
        # get whisper model and preprocessor from transformers
        from transformers import WhisperForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_tag)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_tag)
        self.model = self.model.to(torch.float32)  # use float32 for stability, can be changed to bf16 later
        
        # init error calculator
        from espnet2.legacy.nets.e2e_asr_common import ErrorCalculator
        # get token_list from whisper model
        token_list = self.processor.tokenizer.get_vocab()
        token_list = sorted(token_list, key=token_list.get)
        # we will not use them. init by random
        sym_space, sym_blank = "<space>", "<blank>"
        self.error_calculator = ErrorCalculator(char_list=token_list, sym_space=sym_space, sym_blank=sym_blank, report_cer=True, report_wer=True)

        # for debugging
        def first_nonfinite_hook(name):
            def hook(module, inp, out):
                def check(t, tag):
                    if torch.is_tensor(t):
                        if not torch.isfinite(t).all():
                            raise RuntimeError(
                                f"[NON-FINITE] {name} {tag} "
                                f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
                                f"min={torch.nan_to_num(t).min().item()} "
                                f"max={torch.nan_to_num(t).max().item()}"
                            )
                # check inputs
                if isinstance(inp, (tuple, list)):
                    for k, x in enumerate(inp):
                        check(x, f"in[{k}]")
                else:
                    check(inp, "in")
                # check outputs
                if isinstance(out, (tuple, list)):
                    for k, x in enumerate(out):
                        check(x, f"out[{k}]")
                else:
                    check(out, "out")
            return hook

        handles = []
        for n, m in self.model.named_modules():
            # hook only "real" layers to reduce noise
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.LayerNorm,
                            torch.nn.MultiheadAttention, torch.nn.Embedding)):
                handles.append(m.register_forward_hook(first_nonfinite_hook(n)))

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.model.train()
        else:
            self.model.eval()

        return self
    
    def forward(
        self,
        speech,
        speech_lengths,
        text,
        text_lengths,
        text_ctc,
        text_ctc_lengths,
        text_prev,
        text_prev_lengths,
    ):
        # output requirements: the output should be loss, stats, weights
        # weights is simply the batch size

        # preprocessing and calculate new speech_lengths
        # pad to 30 seconds (3000 frames after processing)
        # devide speech_lengths by 160, and build attention_mask
        # convert speech to list of numpy arrays for processor
        # speech = [s.detach().cpu().numpy() for s in speech]  # list of (T,)
        # processed = self.processor(
        #     speech,
        #     sampling_rate=16000,
        #     return_tensors="pt",
        #     padding=True,
        # )
        # speech = processed.input_features  # (B, D, T')
        # # pad to 30 seconds (3000 frames after processing)
        # speech = torch.nn.functional.pad(speech, (0, max(0, 3000 - speech.size(2))), value=0.0)[:, :, :3000]  # (B, D, 3000)
        # speech_lengths = torch.tensor([min(l // 160, 3000) for l in speech_lengths])  # (B,)
        # attention_mask = (torch.arange(3000).expand(len(speech_lengths), 3000) < speech_lengths.unsqueeze(1)).to(speech.device)  # (B, 3000)

        # transpose back to (B, D, T') for whisper
        speech = speech.transpose(1, 2)  # (B, D, T')
        # pad to 30 seconds (3000 frames after processing)
        speech = torch.nn.functional.pad(speech, (0, max(0, 3000 - speech.size(2))), value=0.0)[:, :, :3000]  # (B, D, 3000)
        attention_mask = torch.arange(3000).expand(len(speech_lengths), 3000).to(speech.device) < speech_lengths.unsqueeze(1)  # (B, 3000)
        
        # make decoder input ids and labels
        decoder_input_ids = text[:, :-1][:,:self.model.config.max_target_positions]  # (B, L-1)
        labels = text[:, 1:][:,:self.model.config.max_target_positions]  # (B, L-1)
        # breakpoint()

        # sanity check 
        # def finite_check(name, x):
        #     if torch.isnan(x).any() or torch.isinf(x).any():
        #         bad = torch.where(~torch.isfinite(x))
        #         raise RuntimeError(f"{name} has NaN/Inf. Example idx: {tuple(b.item() for b in bad[:3])}")
        # finite_check("speech", speech)
        # finite_check("speech_lengths", speech_lengths)
        # assert (speech_lengths > 0).all()
        # assert (speech_lengths <= speech.size(2)).all()
        # vocab = self.model.config.vocab_size
        # def check_labels(name, t):
        #     assert t.dtype == torch.long
        #     bad_neg = (t < 0) & (t != -100)
        #     if bad_neg.any():
        #         raise RuntimeError(f"{name} has invalid negative ids: {t[bad_neg][:10].tolist()}")
        #     bad_hi = t >= vocab
        #     if bad_hi.any():
        #         raise RuntimeError(f"{name} has ids >= vocab: {t[bad_hi][:10].tolist()} (vocab={vocab})")
        # check_labels("labels", labels)
        # check_labels("decoder_input_ids", decoder_input_ids)

        # conv1 = self.model.model.encoder.conv1  # adjust path if needed
        # def check_param(p, name):
        #     ok = torch.isfinite(p).all().item()
        #     print(name, "finite:", ok,
        #         "nan:", torch.isnan(p).any().item(),
        #         "inf:", torch.isinf(p).any().item(),
        #         "min:", torch.nan_to_num(p).min().item(),
        #         "max:", torch.nan_to_num(p).max().item())
        #     return ok
        # check_param(conv1.weight, "conv1.weight")
        # if conv1.bias is not None:
        #     check_param(conv1.bias, "conv1.bias")
        
        output = self.model(input_features=speech, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        # breakpoint()
        loss = output.loss
        acc = th_accuracy(output.logits.reshape(-1, output.logits.size(-1)), labels, ignore_label=50256)        # 50256 is ""
        cer_att, wer_att = None, None
        if not self.training:
            ys_hat = output.logits.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.detach().cpu().numpy(), labels.detach().cpu().numpy())
            cer_att, wer_att = torch.tensor(cer_att), torch.tensor(wer_att)
        stats = {
            "loss": loss,
            "acc": torch.tensor(acc),
            "cer_att": cer_att,
            "wer_att": wer_att,
        }
        return loss, stats, torch.tensor(speech.size(0))

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}
