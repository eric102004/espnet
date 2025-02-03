from espnet2.diar.compressor.abs_compressor import AbsCompressor

class RLECompressor(AbsCompressor):
    def __init__(self, blank_id=-1, max_repeat=-1):
        """
        example: blank_id=31, max_repeat=30
        """
        self.blank_id = blank_id
        self.max_repeat = max_repeat
        self.vocab_size = max_repeat + 2
        assert blank_id == max_repeat + 1, "blank_id should be equal to max_repeat + 1"
    
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
        cur_digit = 0
        repeat_stack = [0]
        for cs in comp_seq:
            # version 1
            s = []
            for i in range(len(cs)):
                if cs[i] in [0, 1] and len(repeat_stack) > 0:
                    s.extend([cur_digit for j in range(sum(repeat_stack))])
                    repeat_stack = []
                    cur_digit = cs[i]
                else:
                    repeat_stack.append(cs[i])
            if len(repeat_stack):
                s.extend([cur_digit for j in range(sum(repeat_stack))])
            # version 2
            #for i in range(0, len(cs), 2):
            #    s.extend([cs[i] for j in range(cs[i+1])])
            seq.append(s)
            seq_length.append(len(s))
        return seq, seq_length
    

class RLECompressor2(AbsCompressor):
    def __init__(self, blank_id=-1, max_repeat=-1):
        """
        example: blank_id=60, max_repeat=30
        """
        self.blank_id = blank_id
        self.max_repeat = max_repeat
        self.vocab_size = self.max_repeat*2 + 1
        assert blank_id == max_repeat*2, "blank_id should be equal to max_repeat*2"
    
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
                    cl.append(self.max_repeat*l[i-1] + (count-1))  # 0:0-29, 1:30-59, blank:60
                    count = 1
            cl.append(self.max_repeat*l[i-1] + (count-1) )
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
            for i in range(len(cs)):
                assert cs[i]<self.blank_id, "compressed label should be less than blank_id"
                b = cs[i] // self.max_repeat
                rep = cs[i] % self.max_repeat + 1
                s.extend([b for j in range(rep)])
            seq.append(s)
            seq_length.append(len(s))
        return seq, seq_length