from espnet2.diar.compressor.abs_compressor import AbsCompressor

class RLECompressor(AbsCompressor):
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