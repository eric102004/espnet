import sentencepiece as spm

vocab_size = 14

# init the BPE model
sp = spm.SentencePieceProcessor(
    model_file=f'bpe{vocab_size}.model'
)

print(sp.get_piece_size())

# load example text
text_file = "train_text.txt"
with open(text_file, 'r') as f:
    text = f.readline().strip()
# convert to list of int
text = [int(i) for i in text]

# conver text to string
text = "".join(str(i) for i in text)
# encode the text
encoded_text = sp.encode(text, out_type=int)
print(f"Encoded text: {encoded_text}")
# (optional) remove leading special token
encoded_text = encoded_text[1:]

# decode the text
decoded_text = sp.decode(encoded_text)
print(f"Decoded text: {decoded_text}")

assert decoded_text == text

decoded_text = [int(i) for i in decoded_text]
print(f"Decoded text (list form): {decoded_text}")