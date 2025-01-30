import pytest
from unittest import mock
from .bpe_compressor import BPECompressor

class TestBPECompressor:
    
    @mock.patch('sentencepiece.SentencePieceProcessor')
    def test_init_with_model_file(self, mock_sp):
        model_file = 'dummy.model'
        compressor = BPECompressor(model_file=model_file)
        mock_sp.assert_called_with(model_file=model_file)
        assert compressor.vocab_size == mock_sp().get_piece_size() + 1
        assert compressor.blank_id == compressor.vocab_size

    """
    @mock.patch('sentencepiece.SentencePieceProcessor')
    @mock.patch('sentencepiece.SentencePieceTrainer.train')
    def test_init_with_data_file(self, mock_train, mock_sp):
        data_file = 'dummy.txt'
        vocab_size = 10
        compressor = BPECompressor(data_file=data_file)
        mock_train.assert_called_with(
            input=data_file,
            model_prefix=f'bpe{vocab_size}',
            vocab_size=vocab_size, 
            character_coverage=1.0,
            model_type='bpe'
        )
        assert compressor.vocab_size == mock_sp().get_piece_size() + 1
        assert compressor.blank_id == compressor.vocab_size
    """

    @mock.patch('sentencepiece.SentencePieceProcessor')
    def test_encode(self, mock_sp):
        mock_sp().encode.return_value = [1, 2, 3]
        compressor = BPECompressor(model_file='dummy.model')
        seq = [[1, 2, 3], [4, 5, 6]]
        comp_seq, comp_seq_length = compressor.encode(seq)
        assert comp_seq == [[1, 2, 3], [1, 2, 3]]
        assert comp_seq_length == [3, 3]

    @mock.patch('sentencepiece.SentencePieceProcessor')
    def test_decode(self, mock_sp):
        mock_sp().decode.return_value = '123'
        compressor = BPECompressor(model_file='dummy.model')
        comp_seq = [[1, 2, 3], [4, 5, 6]]
        seq, seq_length = compressor.decode(comp_seq)
        assert seq == [[1, 2, 3], [1, 2, 3]]
        assert seq_length == [3, 3]