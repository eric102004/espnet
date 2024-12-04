# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import contextmanager
from itertools import permutations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.diar.attractor.abs_attractor import AbsAttractor
from espnet2.diar.decoder.abs_decoder import AbsDecoder
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import to_device

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetDiarizationModel(AbsESPnetModel):
    """Speaker Diarization model

    If "attractor" is "None", SA-EEND will be used.
    Else if "attractor" is not "None", EEND-EDA will be used.
    For the details about SA-EEND and EEND-EDA, refer to the following papers:
    SA-EEND: https://arxiv.org/pdf/1909.06247.pdf
    EEND-EDA: https://arxiv.org/pdf/2005.09921.pdf, https://arxiv.org/pdf/2106.10654.pdf
    """

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        label_aggregator: torch.nn.Module,
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        attractor: Optional[AbsAttractor],
        diar_weight: float = 1.0,
        attractor_weight: float = 1.0,
    ):

        super().__init__()

        self.encoder = encoder
        self.normalize = normalize
        self.frontend = frontend
        self.specaug = specaug
        self.label_aggregator = label_aggregator
        self.diar_weight = diar_weight
        self.attractor_weight = attractor_weight
        self.attractor = attractor
        self.decoder = decoder

        if self.attractor is not None:
            self.decoder = None
        elif self.decoder is not None:
            self.num_spk = decoder.num_spk
        else:
            raise NotImplementedError

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,) default None for chunk interator,
                                     because the chunk-iterator does not
                                     have the speech_lengths returned.
                                     see in
                                     espnet2/iterators/chunk_iter_factory.py
            spk_labels: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert speech.shape[0] == spk_labels.shape[0], (speech.shape, spk_labels.shape)
        batch_size = speech.shape[0]

        # 1. Encoder
        # Use bottleneck_feats if exist. Only for "enh + diar" task.
        bottleneck_feats = kwargs.get("bottleneck_feats", None)
        bottleneck_feats_lengths = kwargs.get("bottleneck_feats_lengths", None)
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, bottleneck_feats, bottleneck_feats_lengths
        )

        if self.attractor is None:
            # 2a. Decoder (baiscally a predction layer after encoder_out)
            pred = self.decoder(encoder_out, encoder_out_lens)
        else:
            # 2b. Encoder Decoder Attractors
            # Shuffle the chronological order of encoder_out, then calculate attractor
            encoder_out_shuffled = encoder_out.clone()
            for i in range(len(encoder_out_lens)):
                encoder_out_shuffled[i, : encoder_out_lens[i], :] = encoder_out[
                    i, torch.randperm(encoder_out_lens[i]), :
                ]
            attractor, att_prob = self.attractor(
                encoder_out_shuffled,
                encoder_out_lens,
                to_device(
                    self,
                    torch.zeros(
                        encoder_out.size(0), spk_labels.size(2) + 1, encoder_out.size(2)
                    ),
                ),
            )
            # Remove the final attractor which does not correspond to a speaker
            # Then multiply the attractors and encoder_out
            pred = torch.bmm(encoder_out, attractor[:, :-1, :].permute(0, 2, 1))
        # 3. Aggregate time-domain labels
        spk_labels, spk_labels_lengths = self.label_aggregator(
            spk_labels, spk_labels_lengths
        )

        # If encoder uses conv* as input_layer (i.e., subsampling),
        # the sequence length of 'pred' might be slighly less than the
        # length of 'spk_labels'. Here we force them to be equal.
        length_diff_tolerance = 2
        length_diff = spk_labels.shape[1] - pred.shape[1]
        if length_diff > 0 and length_diff <= length_diff_tolerance:
            spk_labels = spk_labels[:, 0 : pred.shape[1], :]

        if self.attractor is None:
            loss_pit, loss_att = None, None
            loss, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
        else:
            loss_pit, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
            loss_att = self.attractor_loss(att_prob, spk_labels)
            loss = self.diar_weight * loss_pit + self.attractor_weight * loss_att
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = self.calc_diarization_error(pred, label_perm, encoder_out_lens)

        if speech_scored > 0 and num_frames > 0:
            sad_mr, sad_fr, mi, fa, cf, acc, der = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        else:
            sad_mr, sad_fr, mi, fa, cf, acc, der = 0, 0, 0, 0, 0, 0, 0

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_pit=loss_pit.detach() if loss_pit is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        bottleneck_feats: torch.Tensor,
        bottleneck_feats_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            bottleneck_feats: (Batch, Length, ...): used for enh + diar
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # 4. Forward encoder
            # feats: (Batch, Length, Dim)
            # -> encoder_out: (Batch, Length2, Dim)
            if bottleneck_feats is None:
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
            elif self.frontend is None:
                # use only bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder(
                    bottleneck_feats, bottleneck_feats_lengths
                )
            else:
                # use both frontend and bottleneck feats
                # interpolate (copy) feats frames
                # to match the length with bottleneck_feats
                feats = F.interpolate(
                    feats.transpose(1, 2), size=bottleneck_feats.shape[1]
                ).transpose(1, 2)
                # concatenate frontend LMF feature and bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder(
                    torch.cat((bottleneck_feats, feats), 2), bottleneck_feats_lengths
                )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def pit_loss_single_permute(self, pred, label, length):
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        mask = self.create_length_mask(length, label.size(1), label.size(2))
        loss = bce_loss(pred, label)
        loss = loss * mask
        loss = torch.sum(torch.mean(loss, dim=2), dim=1)
        loss = torch.unsqueeze(loss, dim=1)
        return loss

    def pit_loss(self, pred, label, lengths):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND
        num_output = label.size(2)
        permute_list = [np.array(p) for p in permutations(range(num_output))]
        loss_list = []
        for p in permute_list:
            label_perm = label[:, :, p]
            loss_perm = self.pit_loss_single_permute(pred, label_perm, lengths)
            loss_list.append(loss_perm)
        loss = torch.cat(loss_list, dim=1)
        min_loss, min_idx = torch.min(loss, dim=1)
        loss = torch.sum(min_loss) / torch.sum(lengths.float())
        batch_size = len(min_idx)
        label_list = []
        for i in range(batch_size):
            label_list.append(label[i, :, permute_list[min_idx[i]]].data.cpu().numpy())
        label_permute = torch.from_numpy(np.array(label_list)).float()
        return loss, min_idx, permute_list, label_permute

    def create_length_mask(self, length, max_len, num_output):
        batch_size = len(length)
        mask = torch.zeros(batch_size, max_len, num_output)
        for i in range(batch_size):
            mask[i, : length[i], :] = 1
        mask = to_device(self, mask)
        return mask

    def attractor_loss(self, att_prob, label):
        batch_size = len(label)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # create attractor label [1, 1, ..., 1, 0]
        # att_label: (Batch, num_spk + 1, 1)
        att_label = to_device(self, torch.zeros(batch_size, label.size(2) + 1, 1))
        att_label[:, : label.size(2), :] = 1
        loss = bce_loss(att_prob, att_label)
        loss = torch.mean(torch.mean(loss, dim=1))
        return loss

    @staticmethod
    def calc_diarization_error(pred, label, length):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = length.data.cpu().numpy()

        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
        num_frames = np.sum(length)
        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )


class ESPnetCompressedDiarizationModel(ESPnetDiarizationModel):
    """Speaker Diarization model with compressed target sequence

    If "attractor" is "None", SA-EEND will be used.
    Else if "attractor" is not "None", EEND-EDA will be used.
    For the details about SA-EEND and EEND-EDA, refer to the following papers:
    SA-EEND: https://arxiv.org/pdf/1909.06247.pdf
    EEND-EDA: https://arxiv.org/pdf/2005.09921.pdf, https://arxiv.org/pdf/2106.10654.pdf
    """

    @typechecked
    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        label_aggregator: torch.nn.Module,
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        attractor: Optional[AbsAttractor],
        diar_weight: float = 1.0,
        attractor_weight: float = 1.0,
        compressor: Optional[object],
        blank_id: int = 0,
    ):

        super().__init__(
            frontend, 
            specaug, 
            normalize, 
            label_aggregator, 
            encoder, 
            decoder, 
            attractor, 
            diar_weight, 
            attractor_weight, 
        )

        # add compression model to compress target sequence
        self.compressor = compressor
        self.blank_id = blank_id
        self.ctc_loss = torch.nn.CTCLoss(blank=self.blank_id, reduction='none', zero_infinity=True)
        # TODO: (optional) add beam search

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,) default None for chunk interator,
                                     because the chunk-iterator does not
                                     have the speech_lengths returned.
                                     see in
                                     espnet2/iterators/chunk_iter_factory.py
            spk_labels: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert speech.shape[0] == spk_labels.shape[0], (speech.shape, spk_labels.shape)
        batch_size = speech.shape[0]

        # 1. Encoder
        # Use bottleneck_feats if exist. Only for "enh + diar" task.
        bottleneck_feats = kwargs.get("bottleneck_feats", None)
        bottleneck_feats_lengths = kwargs.get("bottleneck_feats_lengths", None)
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, bottleneck_feats, bottleneck_feats_lengths
        )

        if self.attractor is None:
            # 2a. Decoder (baiscally a predction layer after encoder_out)
            # pred has the shape (batch_size, num_channel, max_len, num_vocabulary)
            pred = self.decoder(encoder_out, encoder_out_lens)
        else:
            # 2b. Encoder Decoder Attractors
            # Shuffle the chronological order of encoder_out, then calculate attractor
            encoder_out_shuffled = encoder_out.clone()
            for i in range(len(encoder_out_lens)):
                encoder_out_shuffled[i, : encoder_out_lens[i], :] = encoder_out[
                    i, torch.randperm(encoder_out_lens[i]), :
                ]
            attractor, att_prob = self.attractor(
                encoder_out_shuffled,
                encoder_out_lens,
                to_device(
                    self,
                    torch.zeros(
                        encoder_out.size(0), spk_labels.size(2) + 1, encoder_out.size(2)
                    ),
                ),
            )
            # Remove the final attractor which does not correspond to a speaker
            # Then multiply the attractors and encoder_out
            # replace bmm with element-wise multiplication, followed by a linear layer
            # encoder_out: (batch_size, max_len, hidden_size)
            # attractor: (batch_size, num_spk + 1, hidden_size)
            # pred has the shape (batch_size, num_channel, max_len, num_vocabulary)
            pred = encoder_out.unsqueeze(1) * attractor[:, :-1, :].unsqueeze(2)
            pred = self.attractor.ctc_lo(pred)
        # 3. Aggregate time-domain labels
        spk_labels, spk_labels_lengths = self.label_aggregator(
            spk_labels, spk_labels_lengths
        )
        spk_labels = spk_labels.permute(0, 2, 1)  # (batch_size, num_channel, max_len)

        # If encoder uses conv* as input_layer (i.e., subsampling),
        # the sequence length of 'pred' might be slighly less than the
        # length of 'spk_labels'. Here we force them to be equal.
        length_diff_tolerance = 2
        length_diff = spk_labels.shape[2] - pred.shape[2]
        if length_diff > 0 and length_diff <= length_diff_tolerance:
            spk_labels = spk_labels[:, :, 0 : pred.shape[2]]

        # use compression model to compress the target sequence
        comp_spk_labels, comp_spk_labels_lengths = self.compress_target_sequence(spk_labels, spk_labels_lengths)

        if self.attractor is None:
            loss_pit, loss_att = None, None
            loss, perm_idx, perm_list, comp_label_perm, comp_label_length_perm = self.pit_loss(
                pred.log_softmax(-1), comp_spk_labels, encoder_out_lens, comp_spk_labels_lengths
            )
        else:
            loss_pit, perm_idx, perm_list, comp_label_perm, comp_label_length_perm = self.pit_loss(
                pred.log_softmax(-1), comp_spk_labels, encoder_out_lens, comp_spk_labels_lengths
            )
            loss_att = self.attractor_loss(att_prob, spk_labels.permute(0, 2, 1))
            loss = self.diar_weight * loss_pit + self.attractor_weight * loss_att

        label_list = []
        for i in range(batch_size):
            label_list.append(spk_labels[i, perm_list[perm_idx[i]], :].data.cpu().numpy())
        label_perm = torch.from_numpy(np.array(label_list)).float()
        
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = self.calc_diarization_error(pred, label_perm, encoder_out_lens)

        if speech_scored > 0 and num_frames > 0:
            sad_mr, sad_fr, mi, fa, cf, acc, der = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        else:
            sad_mr, sad_fr, mi, fa, cf, acc, der = 0, 0, 0, 0, 0, 0, 0

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_pit=loss_pit.detach() if loss_pit is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
    
    def pit_loss_single_permute(self, pred, label, length, label_length):
        """
        replace bce loss with ctc loss
        Args:
            pred: (batch_size, num_channel, max_len, num_vocabulary)
            label: compressed label (batch_size, num_channel, max_label_len)
            length: compressed label length (batch_size, num_channel)
            label_length: (batch_size, num_channel)
        Returns:
            loss: (batch_size, 1)
        """
        # change the shape of pred to (max_len, batch_size * num_channel, num_vocabulary)
        bs = pred.size(0)
        pred = pred.reshape(-1, pred.size(2), pred.size(3))  # (batch_size * num_channel, max_len, num_vocabulary)
        pred = pred.permute(1, 0, 2)  # (max_len, batch_size * num_channel, num_vocabulary)
        # change the shape of label to (batch_size * num_channel, max_label_len)
        label = label.reshape(-1, label.size(2)) 
        # change the shape of length to (batch_size * num_channel)
        length = length.reshape(-1)
        # change the shape of label_length to (batch_size * num_channel)
        label_length = label_length.reshape(-1)
        loss = self.ctc_loss(pred, label, length, label_length)
        # change the shape of loss to (batch_size, num_channel)
        loss = loss.reshape(bs, -1)
        # calculate the mean of loss over num_channel
        loss = torch.mean(loss, dim=1, keepdim=True)
        return loss
    
    def pit_loss(self, pred, label, lengths, label_lengths):
        """
        Args:
            pred: (batch_size, num_channel, max_len, num_vocabulary)
            label: compressed label (batch_size, num_channel, max_label_len)
            lengths: (batch_size, num_channel)
            label_lengths: (batch_size, num_channel)
        """
        num_output = label.size(1)
        permute_list = [np.array(p) for p in permutations(range(num_output))]
        loss_list = []
        for p in permute_list:
            label_perm, label_lengths = label[:, p, :], label_lengths[:,p]
            loss_perm = self.pit_loss_single_permute(pred, label_perm, lengths, label_lengths)
            loss_list.append(loss_perm)
        loss = torch.cat(loss_list, dim=1)
        min_loss, min_idx = torch.min(loss, dim=1)
        loss = torch.sum(min_loss)
        batch_size = len(min_idx)
        label_list = []
        label_lengths_list = []
        for i in range(batch_size):
            label_list.append(label[i, permute_list[min_idx[i]], :].data.cpu().numpy())
            label_lengths_list.append(label_lengths[i, permute_list[min_idx[i]]].data.cpu().numpy())
        label_permute = torch.from_numpy(np.array(label_list)).float()
        label_lengths_permute = torch.from_numpy(np.array(label_lengths_list))
        return loss, min_idx, permute_list, label_permute, label_lengths_permute
    
    def compress_target_sequence(self, label, label_lengths):
        # compress the target sequence using compression model
        """
        Args:
            label has the shape (batch_size, num_channel, max_label_len)
            label_lengths has the shape (batch_size, num_channel)
        Returns:
            comp_label has the shape (batch_size, num_channel, max_comp_len)
            comp_label_lengths has the shape (batch_size, num_channel)
        """
        bs = label.size(0)
        n_channel = label.size(1)
        label = label.reshape(-1, label.size(2))
        label_lengths = label_lengths.reshape(-1)
        # convert label into list[list[int]]
        label = [l[:label_lengths[i]].tolist() for i, l in enumerate(label)]
        comp_label, comp_label_lengths = self.compressor.encode(label, label_lengths)
        # zero pad comp_label and convert it into tensor
        comp_label = self.align_length_and_to_tensor(comp_label, comp_label_lengths)
        comp_label = comp_label.reshape(bs, n_channel, comp_label.size(1))
        comp_label_lengths = torch.tensor(comp_label_lengths).reshape(bs, n_channel)
        return comp_label, comp_label_lengths
    
    @staticmethod
    def calc_diarization_error(self, pred, label, length):
        """
        Args:
            pred has the shape (batch_size, num_channel, max_len, num_vocabulary)
            label has the shape (batch_size, num_channel, max_len)
            length has the shape (batch_size, num_channel)
        """
        # ctc decode pred into comp_pred_seq
        # comp_pred_seq is list[list[list[int]]] and has the shape (batch_size, num_channel, max_comp_len)
        comp_pred_seq = self.ctc_decode(pred)

        # decode comp_pred_seq into pred_seq (the output binary sequence)
        # pred_seq has the shape (batch_size, max_len, num_channel)
        pred_seq = self.decompress_pred_seq(comp_pred_seq, length) # (batch_size, num_channel, max_len)
        pred_seq = pred_seq.permute(0, 2, 1)  # (batch_size, max_len, num_channel)
        label = label.permute(0, 2, 1)  # (batch_size, max_len, num_channel)

        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        pred_np = pred.data.cpu().numpy().astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = length.data.cpu().numpy()

        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
        num_frames = np.sum(length)
        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )

    def ctc_decode(self, pred):
        """
        decode the ctc output into sequence
        Args:
            pred has the shape (batch_size, num_channel, max_len, num_vocabulary)
        Returns:
            pred_seqs is list[list[list[int]]] and has the shape (batch_size, num_channel, pred_seq_len)
        """
        # decode the ctc output into sequence
        token_int = torch.argmax(pred, dim=-1)    # (batch_size, num_channel, max_len)
        #token_int = token_int.data.cpu().tolist()
        pred_seqs = []
        for token_int_channel in token_int:
            pred_seqs_channel = []
            for tokens in token_int_channel:
                pred_seq = torch.unique_consecutive(tokens).tolist()
                pred_seq = [token for token in pred_seq if token != self.blank_id]
                pred_seqs_channel.append(pred_seq)
            pred_seqs.append(pred_seqs_channel)
        return pred_seqs

    def decompress_pred_seq(self, comp_pred_seq, ref_decompressed_length):
        """
        decode the compressed sequence into sequence
        Args:
            comp_pred_seq (list[list[list[int]]]) has the shape (batch_size, num_channel, [comp_len])
            ref_decompredded_length has the shape (batch_size, )
        Returns:
            pred_seq has the shape (batch_size, num_channel, max_len)
        """
        # decode the compressed sequence into sequence
        # step 1 : reshape the comp_pred_seq into (batch_size * num_channel, max_comp_len)
        new_comp_pred_seq = []
        for i in range(len(comp_pred_seq)):
            for j in range(len(comp_pred_seq[i])):
                new_comp_pred_seq.append(comp_pred_seq[i][j])
        comp_pred_seq = new_comp_pred_seq
        # step 2 : decompress the comp_pred_seq into pred_seq using compression model
        pred_seq, _ = self.compressor.decode(comp_pred_seq)
        # step 3 : zero pad pred_seq and convert it into tensor
        pred_seq = self.align_length_and_to_tensor(pred_seq, ref_decompressed_length)
        return pred_seq
        

    def align_length_and_to_tensor(self, seq, ref_length):
        """
        align the length of seq and length, and convert seq into tensor
        Args:
            seq (list[list[int]]) has the shape (batch_size * num_channel, [each sequence's length])
            ref_length has the shape (batch_size, )
        Returns:
            seq has the shape (batch_size, num_channel, max(length))
        """
        batch_size = ref_length.size(0)
        num_channel = len(seq) // batch_size
        # truncate the length of seq with length
        for i in range(batch_size):
            for j in range(i*num_channel, (i+1)*num_channel):
                seq[j] = seq[j][:ref_length[i]]
        # zero pad seq with length.max()
        max_length = ref_length.max()
        for i in range(batch_size):
            for j in range(len(seq[i]), max_length):
                seq[i].append(0)
        # convert seq into tensor
        seq = torch.tensor(seq).reshape(batch_size, num_channel, max_length)
        return seq
        
        
        
class BPECompressionModel:
    def __init__(self, vocab_file):
        self.vocab_dict, self.inv_vocab_dict = self.load_vocab(vocab_file)
        # TODO: craeate string2list mapping
        
    def load_vocab(self, vocab_file):
        """
        example content of vocab file
        0 0
        1 1 
        00 2
        11 3
        """
        with open(vocab_file, 'r') as f:
            vocab_dict = {}
            inverse_vocab_dict = {}
            for line in f:
                token, idx = line.strip().split()
                vocab_dict[token] = int(idx)
                inverse_vocab_dict[int(idx)] = token
        return vocab_dict, inverse_vocab_dict
    
    def encode(self, seq, *args, **kwargs):
        # compress the label sequence
        """
        seq (list[list[int]]) has the shape (num_seq, [decomp_len])
        comp_seq (list[list[int]]) has the shape (num_seq, [comp_len])
        """
        # TODO: operate on the string level
        comp_seq = []
        for s in seq:
            cs = [self.vocab_dict[c] for c in s]
            comp_seq.append(cs)
        return comp_seq
    
    def decode(self, comp_seq, *args, **kwargs):
        # decode the compressed label sequence
        """
        comp_seq (list[list[int]]) has the shape (num_seq, [comp_len])
        seq (list[list[int]]) has the shape (num_seq, [decomp_len])
        """
        # TODO: operate on the string level
        seq = []
        for cs in comp_seq:
            s = [self.inv_vocab_dict[c] for c in cs]
            seq.append(s)
        return seq

class RLECompressionModel:
    def __init__(self, max_repeat=30):
        self.max_repeat = max_repeat
    
    def encode(self, label, *args, **kwargs):
        # compress the label sequence using run-length encoding
        """
        Args:
            label (list[list[int]]) has the shape (num_seq, [decomp_len])
        Returns:
            comp_label (list[list[int]]) has the shape (num_seq, [comp_len])
            comp_label_length (list[int]) has the shape (num_seq)
        """
        comp_label = []
        comp_label_length = []
        for l in label:
            cl = []
            count = 1
            for i in range(1, len(l)):
                if l[i] == l[i-1] and count < self.max_repeat:
                    count += 1
                else:
                    cl.append(l[i-1])
                    cl.append(count)
                    count = 1
            cl.append(l[-1])
            cl.append(count)
            comp_label.append(cl)
            comp_label_length.append(len(cl))
        return comp_label, comp_label_length
    
    def decode(self, comp_seq, *args, **kwargs):
        # decode the compressed label sequence
        """
        Args:
            comp_seq (list[list[int]]) has the shape (num_seq, [comp_len])
        Returns:
            seq (list[list[int]]) has the shape (num_seq, [decomp_len])
            seq_length (list[int]) has the shape (num_seq)
        """
        seq = []
        seq_length = []
        for cs in comp_seq:
            s = []
            for i in range(0, len(cs), 2):
                s.extend([cs[i] for j in range(cs[i+1])])
            seq.append(s)
            seq_length.append(len(s))
        return seq, seq_length