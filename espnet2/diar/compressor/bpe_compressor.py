from espnet2.diar.compressor.abs_compressor import AbsCompressor

class BPECompressor(AbsCompressor):
    def __init__(self, vocab_file):
        self.vocab_dict, self.inv_vocab_dict = self.load_vocab(vocab_file)
        self.vocab_size = len(self.vocab_dict)
        self.blank_id = self.vocab_dict["<b>"]
        
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
        Args:
            seq (list[list[int]]) has the shape (num_seq, [decomp_len])
        Returns:
            comp_seq (list[list[int]]) has the shape (num_seq, [comp_len])
        """
        comp_seq = []
        for subseq in seq:
            compressed = []
            i = 0
            while i < len(subseq):
                longest_match = 1
                for j in range(2, min(len(subseq) - i + 1, 3)):  # Look for matches up to length 2
                    token = ''.join(map(str, subseq[i:i+j]))
                    if token in self.vocab_dict:
                        longest_match = j
                
                if longest_match > 1:
                    token = ''.join(map(str, subseq[i:i+longest_match]))
                    compressed.append(self.vocab_dict[token])
                else:
                    compressed.append(self.vocab_dict[str(subseq[i])])
                
                i += longest_match
            
            comp_seq.append(compressed)
        
        return comp_seq
    
    def decode(self, comp_seq, *args, **kwargs):
        # decode the compressed label sequence
        """
        Args:
            comp_seq (list[list[int]]) has the shape (num_seq, [comp_len])
        Returns:
            seq (list[list[int]]) has the shape (num_seq, [decomp_len])
        """
        seq = []
        for subseq in comp_seq:
            decompressed = []
            for token_id in subseq:
                if token_id == self.blank_id:
                    continue
                token = self.inv_vocab_dict[token_id]
                if token.isdigit():
                    decompressed.extend(map(int, token))
                else:
                    decompressed.append(int(token))
            
            seq.append(decompressed)
        
        return seq
    
    def train_bpe(self, data):
        """
        Args:

        Returns:

        """
        self.