class Vocab(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        
        # Fixed vocabulary
        vocab_tokens = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>', 'A', 'T', 'C', 'G']
        
        # Build the vocab dictionaries
        for idx, token in enumerate(vocab_tokens):
            self.word2idx[token] = idx
            self.idx2word.append(token)
        
        # Set the size of the vocabulary
        self.size = len(self.word2idx)
        
        # Define special tokens by their index
        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']
        self.nspecial = 5  # Number of special tokens