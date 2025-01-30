import sentencepiece as spm

vocab_size = 14
data_file = "train_text.txt"

# train the BPE model. It will generate the model file bpe<vocab_size>.model
spm.SentencePieceTrainer.train(
    input=data_file,
    model_prefix=f'bpe{vocab_size}',
    vocab_size=vocab_size, 
    character_coverage=1.0,
    model_type='bpe'
)

# load the BPE model
sp = spm.SentencePieceProcessor(
    model_file=f'bpe{vocab_size}.model'
)

print('finished training BPE model')