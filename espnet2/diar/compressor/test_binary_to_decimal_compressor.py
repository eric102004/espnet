import pytest
from espnet2.diar.compressor.binary_to_decimal_compressor import BinaryToDecimalCompressor

# FILE: <espnet_root>/espnet2/diar/compressor/test_binary_to_decimal_compressor.py


class TestBinaryToDecimalCompressor:
    
    def test_init_valid(self):
        compressor = BinaryToDecimalCompressor(blank_id=4, compression_rate=2)
        assert compressor.compression_rate == 2
        assert compressor.vocab_size == 5
    
    def test_init_invalid_blank_id(self):
        with pytest.raises(AssertionError, match="blank_id should be equal to 2**compression_rate"):
            BinaryToDecimalCompressor(blank_id=3, compression_rate=2)
    
    def test_init_valid_different_rate(self):
        compressor = BinaryToDecimalCompressor(blank_id=8, compression_rate=3)
        assert compressor.compression_rate == 3
        assert compressor.vocab_size == 9
    
    def test_init_invalid_blank_id_different_rate(self):
        with pytest.raises(AssertionError, match="blank_id should be equal to 2**compression_rate"):
            BinaryToDecimalCompressor(blank_id=7, compression_rate=3)
    
    def test_encode(self):
        compressor = BinaryToDecimalCompressor(blank_id=4, compression_rate=2)
        label = [[1, 0, 1, 1], [0, 1]]
        expected_comp_label = [[2, 3], [1]]
        expected_comp_label_length = [2, 1]
        comp_label, comp_label_length = compressor.encode(label)
        assert comp_label == expected_comp_label
        assert comp_label_length == expected_comp_label_length
    
    def test_decode(self):
        compressor = BinaryToDecimalCompressor(blank_id=4, compression_rate=2)
        comp_seq = [[2, 3], [1]]
        expected_seq = [[1, 0, 1, 1], [0, 1]]
        expected_seq_length = [4, 2]
        seq, seq_length = compressor.decode(comp_seq)
        assert seq == expected_seq
        assert seq_length == expected_seq_length
    
    def test_decode_with_ref_seq_length(self):
        compressor = BinaryToDecimalCompressor(blank_id=4, compression_rate=2)
        comp_seq = [[2, 3], [1]]
        ref_seq_length = [3, 1]
        expected_seq = [[1, 0, 1], [0]]
        seq, seq_length = compressor.decode(comp_seq, ref_seq_length)
        assert seq == expected_seq
        assert seq_length == ref_seq_length