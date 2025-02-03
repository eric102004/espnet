from espnet2.diar.compressor.abs_compressor import AbsCompressor

import sentencepiece as spm

class BPECompressor(AbsCompressor):
    def __init__(self, model_file=None, data_file=None, bpe_vocab_size=10):
        if model_file is not None:
            self.load_model(model_file)
        else:
            assert data_file is not None, "Either model_file or data_file must be provided"
            self.train_bpe(data_file, bpe_vocab_size)
        self.vocab_size = self.sp.get_piece_size() - 4 + 1 # (bpe_vocab_size - 4) + blank id
        self.blank_id = self.vocab_size - 1 
        
        
    def load_model(self, model_file):
        self.sp = spm.SentencePieceProcessor(
            model_file=model_file
        )
    
    def encode(self, seq, *args, **kwargs):
        # compress the label sequence
        """
        Args:
            seq (list[list[int]]) has the shape (num_seq, [decomp_len])
        Returns:
            comp_seq (list[list[int]]) has the shape (num_seq, [comp_len])
            comp_seq_length (list[int]) has the shape (num_seq)
        """
        # convert seq in list of str
        seq = ["".join(str(int(i)) for i in subseq) for subseq in seq]
        # encode using spm
        comp_seq = [self.sp.encode(s, out_type=int) for s in seq]
        # postprocess: map vocab from sentence piece model to espnet model
        comp_seq = [[i-3 for i in s] for s in comp_seq]
        # check all the ids are in the vocab
        comp_seq_length = [len(s) for s in comp_seq]
        return comp_seq, comp_seq_length
    
    def decode(self, comp_seq, *args, **kwargs):
        # decode the compressed label sequence
        """
        Args:
            comp_seq (list[list[int]]) has the shape (num_seq, [comp_len])
        Returns:
            seq (list[list[int]]) has the shape (num_seq, [decomp_len])
            seq_length (list[int]) has the shape (num_seq)
        """
        # preprocess : remove blank id and map them to vocab of sentence piece model
        comp_seq = [[i+3 for i in s if i!=self.blank_id] for s in comp_seq]
        # decode using spom
        seq = [self.sp.decode(s) for s in comp_seq]
        # convert seq to list of list of int
        try:
            seq = [[int(i) for i in s] for s in seq]
        except:
            print(seq)
            raise ValueError("Error decoding sequence")
        seq_length = [len(s) for s in seq]
        return seq, seq_length
    
    def train_bpe(self, data_file, vocab_size=10):
        """
        Args:
            data_file (str): path to the data file
            vocab_size (int): vocabulary size for the BPE model
        Returns:
            None
        """
        # train the BPE model. It will generate the model file bpe<vocab_size>.model
        spm.SentencePieceTrainer.train(
            input=data_file,
            model_prefix=f'bpe{vocab_size}',
            vocab_size=vocab_size, 
            character_coverage=1.0,
            model_type='bpe'
        )

        # load the trained model
        self.sp = spm.SentencePieceProcessor(model_file=f'bpe{vocab_size}.model')

        return