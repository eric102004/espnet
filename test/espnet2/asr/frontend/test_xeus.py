import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.frontend.xeus import XEUSFrontend

is_torch_1_8_plus = V(torch.__version__) >= V("1.8.0")

xeus_path = "../../../../xeus/xeus_checkpoint.pth"
device="cuda"

@pytest.mark.skipif(not is_torch_1_8_plus, reason="Not supported")
def test_frontend_init():
    frontend = XEUSFrontend(
        fs=16000,
        frontend_conf=dict(ckpt_path=xeus_path, device=device),
    )
    assert frontend.frontend_type == "xeus"
    assert frontend.output_size() > 0


@pytest.mark.skipif(not is_torch_1_8_plus, reason="Not supported")
def test_frontend_output_size():
    frontend = XEUSFrontend(
        fs=16000,
        frontend_conf=dict(ckpt_path=xeus_path, device=device),
    )

    wavs = torch.randn(2, 1600).to(device)
    lengths = torch.LongTensor([1600, 800]).to(device)
    feats, _ = frontend(wavs, lengths)
    assert feats.shape[-1] == frontend.output_size()


@pytest.mark.skipif(not is_torch_1_8_plus, reason="Not supported")
@pytest.mark.parametrize(
    "fs, frontend_conf, multilayer_feature, layer",
    [
        (16000, dict(ckpt_path=xeus_path, device=device), True, -1),
        (16000, dict(ckpt_path=xeus_path, device=device), False, -1),
        (16000, dict(ckpt_path=xeus_path, device=device, tile_factor=1), False, -1),
        (16000, dict(ckpt_path=xeus_path, device=device), False, 0),
    ],
)
def test_frontend_backward(fs, frontend_conf, multilayer_feature, layer):
    frontend = XEUSFrontend(
        fs=fs,
        frontend_conf=frontend_conf,
        multilayer_feature=multilayer_feature,
        layer=layer,
    )
    wavs = torch.randn(2, 1600, requires_grad=True).to(device)
    lengths = torch.LongTensor([1600, 800]).to(device)
    feats, f_lengths = frontend(wavs, lengths)
    feats.sum().backward()