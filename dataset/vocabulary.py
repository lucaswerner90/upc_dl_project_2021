import nltk

class Vocabulary():
    """
    Class for building and handling vocabulary in dataset
    """
    def __init__(self):
        #initialize known tokens
        self.index_to_word = {0: "<PAD>", 1:"<START>", 2:"<END>",3:"<UNK>"}

        self.word_to_index = {v:k for k, v in self.index_to_word.items()}

    def __len__(self):
        return len(self.word_to_index)

    @staticmethod
    def tokenize(text):
        return nltk.word_tokenize(text.lower())
    
    def build_vocabulary(self,sentence_list,reduce=True, max_size=5000):
        #download punctuation
        nltk.download("punkt")
        freq = {}

        #Build vocabulary set on word frequency
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                if word in freq:
                   freq[word] += 1
                else:
                    freq[word] = 1         
                
        #Sort by frequency
        items=list(freq.items())
        items.sort(key=lambda x:x[1],reverse=True)

        #if reduce is True, keep only max_size words in freq order
        if reduce: 
            items=items[:max_size]
            
        idx = len(self.index_to_word) #starting idx

        #Fill up itow to add it to initial dictionary
        for k,v in items:
            self.index_to_word[idx]=k
            idx+=1
        self.word_to_index = {v:k for k, v in self.index_to_word.items()}
        
    #to assign token from vocab from text
    def numericalize(self,text):
        tok_text = self.tokenize(text)
        return [self.word_to_index[token] if token in self.word_to_index else self.word_to_index['<UNK>'] for token in tok_text]

    #to reconstruct caption
    #TODO: Pending to concat caption
    def generate_caption(self,vec):
        return ' '.join([self.index_to_word[token] if token in self.index_to_word else '<UNK>' for token in vec.tolist()])