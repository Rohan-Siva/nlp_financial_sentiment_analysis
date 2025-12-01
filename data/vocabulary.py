import collections

class Vocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = collections.Counter()
        
    def build_vocabulary(self, tokenized_sentences):

        for tokens in tokenized_sentences:
            self.word_counts.update(tokens)
            
        idx = 2
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
    def convert_tokens_to_ids(self, tokens):

        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
  
        return [self.idx2word.get(idx, '<UNK>') for idx in ids]
    
    def __len__(self):
        return len(self.word2idx)
