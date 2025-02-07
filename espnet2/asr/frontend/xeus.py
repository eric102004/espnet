import copy
import logging
from typing import Optional, Tuple, Union, List

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend

from espnet2.tasks.ssl import SSLTask


class XEUSFrontend(AbsFrontend):
    """XEUS frontend structure for ASR with weighted sum of multilayer features."""

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        multilayer_feature: bool = False,
        layer: int = -1,
        use_flash_attn: bool = False,
    ):
        try:
            import s3prl
            from s3prl.nn import Featurizer
        except Exception as e:
            print("Error: S3PRL is not properly installed.")
            print("Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done")
            raise e

        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        assert fs == 16000, "XEUS frontend only supports 16kHz sampling rate."

        # init xeus upstream
        xeus_model, xeus_train_args = SSLTask.build_model_from_file(
            None,
            frontend_conf.get("ckpt_path"),
            frontend_conf.get("device"),   # TODO: check how to set this
        )
        if use_flash_attn:
            for layer in xeus_model.encoder.encoders:
                layer.use_flash_attn=True
        upstream = xeus_model
        upstream.eval()
        # set required properties for featurizer
        upstream.num_layers = 19
        upstream.hidden_sizes = [1024]
        upstream.downsample_rates = [1]
        
        if layer != -1:
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when specific layer used"
        else:
            layer_selections = None
        # init featurizer (using s3prl featurizer or implemnt a similar one)
        featurizer = Featurizer(upstream, layer_selections=layer_selections)

        self.multilayer_feature = multilayer_feature
        self.layer = layer
        self.upstream, self.featurizer = upstream, featurizer
        self.pretrained_params = copy.deepcopy(self.upstream.state_dict())
        self.frontend_type = "xeus"
        self.hop_length = self.featurizer.downsample_rate
        self.tile_factor = frontend_conf.get("tile_factor", 1)
        self.use_flash_attn = use_flash_attn
        self.fs = fs

    def _tile_representations(self, feature):
        """Tile up the representations by `tile_factor`.

        Input - sequence of representations
                shape: (batch_size, seq_len, feature_dim)

        Output - sequence of tiled representations
                 shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(feature.shape) == 3
        ), "Input argument `feature` has invalid shape: {}".format(feature.shape)
        tiled_feature = feature.repeat(1, 1, self.tile_factor)
        tiled_feature = tiled_feature.reshape(
            feature.size(0), feature.size(1) * self.tile_factor, feature.size(2)
        )
        return tiled_feature
    
    def output_size(self) -> int:
        return self.featurizer.output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        feats, feats_lens = self.upstream_forward(input, input_lengths)

        if self.layer != -1:
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        if self.tile_factor != 1:
            feats = self._tile_representations(feats)
        
        return feats, feats_lens
    
    def reload_pretrained_parameters(self):
        self.upstream.load_state_dict(self.pretrained_params)
        logging.info("Pretrained XEUS frontend model parameters reloaded!")

    def upstream_forward(
        self, wavs: torch.Tensor, wavs_len: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        modified from https://github.com/s3prl/s3prl/blob/b78ff25d33f69b720519630f682fe7daaacd4698/s3prl/nn/upstream.py#L181-L231
        """
        MIN_SECOND = 0.05
        if wavs.dim() == 3:
            wavs = wavs.squeeze(-1)

        original_wavs_len = wavs_len
        if max(original_wavs_len) < MIN_SECOND * self.fs:
            padded_samples = int(MIN_SECOND * self.fs) - max(original_wavs_len)
            wavs = torch.cat(
                (wavs, wavs.new_zeros(wavs.size(0), padded_samples)),
                dim=1,
            )
            wavs_len = wavs_len + padded_samples

        #wavs_list = []
        #for wav, wav_len in zip(wavs, wavs_len):
        #    wavs_list.append(wav[:wav_len])
        use_mask = self.training
        if self.use_flash_attn:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                hidden_states, out_lens, _, _ = self.upstream.encode(wavs, wavs_len, use_mask=use_mask, use_final_output=False)
        else:
                hidden_states, out_lens, _, _ = self.upstream.encode(wavs, wavs_len, use_mask=use_mask, use_final_output=False)
        
        # repeat out_lens by num_layers times
        out_lens = [out_lens] * self.upstream.num_layers
        
        assert isinstance(hidden_states, (list, tuple))
        assert (
            len(hidden_states) == self.upstream.num_layers
        ), f"{len(hidden_states)}, {self.upstream.num_layers}"

        return hidden_states, out_lens

        #max_wav_len = int(max(wavs_len))
        #all_hs = []
        #all_lens = []
        #for h, stride in zip(hidden_states, self.downsample_rates):
        #    expected_max_h_len = len(range(0, max_wav_len, stride))
        #    h = self._match_length(h, expected_max_h_len)
        #    assert h.size(1) == expected_max_h_len

        #    h_len = torch.div(original_wavs_len - 1, stride, rounding_mode="floor") + 1
        #    h = h[:, : max(h_len), :]
        #    if self.normalize:
        #        h = F.layer_norm(h, h.shape[-1:])

        #    all_hs.append(h)
        #    all_lens.append(h_len)

        #return all_hs, all_lens
    

