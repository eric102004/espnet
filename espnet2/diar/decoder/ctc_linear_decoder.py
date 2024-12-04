import torch

from espnet2.diar.decoder.abs_decoder import AbsDecoder


class CTCLinearDecoder(AbsDecoder):
    """Linear decoder for speaker diarization"""

    def __init__(
        self,
        encoder_output_size: int,
        num_spk: int = 2,
        vocab_size: int = 0,
        dropout_rate: float = 0.0 
    ):
        super().__init__()
        self._num_spk = num_spk
        self.ctc_lo = torch.nn.Linear(encoder_output_size, num_spk * vocab_size)
        self.dropout_layer = torch.nn.Dropout(p=dropout_rate)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        Output:
            output (torch.Tensor): [Batch, num_spk, T, vocab_size]
        """

        output = self.ctc_lo(self.dropout_layer(input))    # [Batch, T, num_spk * vocab_size]
        output = output.view(output.size(0), output.size(1), self.num_spk, -1)    # [Batch, T, num_spk, vocab_size]
        output = output.permute(0, 2, 1, 3)    # [Batch, num_spk, T, vocab_size]

        return output

    @property
    def num_spk(self):
        return self._num_spk
