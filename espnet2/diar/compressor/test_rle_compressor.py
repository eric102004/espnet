import pytest
from unittest import mock
from espnet2.diar.compressor.rle_compressor import RLECompressor

@pytest.fixture
def rle_compressor():
    return RLECompressor()

def test_encode(rle_compressor):
    label = [[1, 1, 2, 2, 2, 3, 3, 3, 3]]
    expected_comp_label = [[1, 2, 2, 3, 3, 3]]
    expected_comp_label_length = [6]
    
    comp_label, comp_label_length = rle_compressor.encode(label)
    
    assert comp_label == expected_comp_label
    assert comp_label_length == expected_comp_label_length

def test_decode(rle_compressor):
    comp_seq = [[1, 2, 2, 3, 3, 3]]
    expected_seq = [[1, 1, 2, 2, 2, 3, 3, 3, 3]]
    expected_seq_length = [9]
    
    seq, seq_length = rle_compressor.decode(comp_seq)
    
    assert seq == expected_seq
    assert seq_length == expected_seq_length