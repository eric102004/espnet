from espnet2.diar.compressor.abs_compressor import AbsCompressor
    

class BinaryToDecimalCompressor(AbsCompressor):
    def __init__(self, blank_id=4, compression_rate=2):
        """
        Initialize the BinaryToDecimalCompressor.
        assert blank_id == 2**compression_rate, f"blank_id should be equal to 2**compression_rate, but got blank_id={blank_id} and 2**compression_rate={2**compression_rate}"
        Args:
            blank_id (int): The ID used for blank tokens, should be equal to 2**compression_rate.
            compression_rate (int): The rate of compression, determines the size of the vocabulary.
        """
        self.compression_rate = compression_rate
        self.vocab_size = 2**compression_rate + 1
        assert blank_id == 2**compression_rate, "blank_id should be equal to 2**compression_rate"
    
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
            for i in range(0, len(l), self.compression_rate):
                # padding 
                if len(l) - i < self.compression_rate:
                    l.extend([0 for j in range(self.compression_rate - len(l) + i)])
                comp = 0
                for j in range(self.compression_rate):
                    comp += l[i+j] * 2**(self.compression_rate - j - 1)
                cl.append(comp)
            comp_label.append(cl)
            comp_label_length.append(len(cl))
        return comp_label, comp_label_length
    
    def decode(self, comp_seq, ref_seq_length=None, *args, **kwargs):
        # decode the compressed label sequence
        """
        Args:
            comp_seq (list[list[int]]) has the shape (num_seq, [comp_len])
        Returns:
            seq (list[list[int]]) has the shape (num_seq, [decomp_len])
            seq_length (list[int]) has the shape (num_seq)
        """
        seq = []
        if ref_seq_length is None:
            seq_length = []
        for k, cs in enumerate(comp_seq):
            s = []
            for i in range(len(cs)):
                # convert cs[i] in binary
                for j in range(self.compression_rate):
                    s.append((cs[i] >> (self.compression_rate - j - 1)) & 1)
            if ref_seq_length is None:
                seq_length.append(len(s))
            else:
                s = s[:ref_seq_length[k]]
            seq.append(s)
        if ref_seq_length is None:
            return seq, seq_length
        else:
            return seq, ref_seq_length