from espnet2.diar.compressor.abs_compressor import AbsCompressor

class BPECompressionModel(AbsCompressor):
    def __init__(self, vocab_file):
        self.vocab_dict, self.inv_vocab_dict = self.load_vocab(vocab_file)
        # TODO: craeate string2list mapping
        
    def load_vocab(self, vocab_file):
        """
        example content of vocab file
        <b> 0
        0 1
        1 2 
        00 3
        11 4
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