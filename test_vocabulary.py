import pandas as pd

captions = pd.read_csv(open("/Users/Adri/Desktop/Postgrau DL/datasets/archive/captions.txt",'r'))

sentences=captions['caption']

def compute_vocabulary_freq(sentences):
  vocabulary = {}
  number_of_words = 0
  for sent in sentences:
    for token in sent.split(' '):
      if token in vocabulary:
        vocabulary[token] +=1
      else:
        vocabulary[token] = 1
      number_of_words += 1
  return vocabulary, number_of_words

vocabulary , number_of_words = compute_vocabulary_freq(sentences)
print("Len of training vocabulary is: ",len(vocabulary),"\n","Total number of words are: ",number_of_words)

def compute_vocabulary(train_sents):
  vocabulary = {'<unk>':99999999}
  for sent in train_sents:
    for token in sent:
      if token in vocabulary:
        vocabulary[token] += 1
      else:
        vocabulary[token] = 1
      
  return vocabulary

def coverage(split,voc):
  total = 0.0
  unk = 0.0
  for sent in split:
    for token in sent:
      if not  token in voc:
          unk += 1.0
      total += 1.0
  return 1.0 - (unk/total)


def voc_stats(split,voc):
  print('**** VOCABULARY ***')
  print('* Unique words', len(voc))
  print('* Coverage', coverage(split,voc))

import nltk
nltk.download('stopwords') #Numbers, unused words
nltk.download('wordnet') #dictionary of words and relations and different languages
nltk.download('punkt') #punctuation

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer    
from nltk.tokenize import TweetTokenizer
import numpy as np


stop_words = list(stopwords.words('english')) #About 150 stopwords
lemmatizer = WordNetLemmatizer()
#TOKENIZATION

def tokenize_sentence(sentence):
  return nltk.word_tokenize(sentence)

tok_sentences = [tokenize_sentence(s) for s in sentences]
print('After TOKENIZATION\n')
tok_voc = compute_vocabulary(tok_sentences)
print('* Unique words', len(tok_voc))

#Casing and stop words removal
def preprocess_sentence(sentence):
    return  [word.lower() for word in sentence if not word in stop_words]

tokc_sentences = [preprocess_sentence(s) for s in tok_sentences]

print('After CASING AND STOP WORD Removal\n')
tok_voc_c = compute_vocabulary(tokc_sentences)
print('* Unique words', len(tok_voc_c))

print('Now with Gensim processing')
import gensim
gensim_sentences = []
for s in sentences:
  gensim_sentences.append(gensim.utils.simple_preprocess(s))

gensim_sentences=list(gensim_sentences)

#gensim_sentences = list([gensim.utils.simple_preprocess(s) for s in sentences])

print('After gensim processing\n')
gen_voc = compute_vocabulary(gensim_sentences)
print('* Unique words', len(gen_voc))

#TRAIN WORD2VEC
print("Training word2vec")
w2v_model = gensim.models.Word2Vec(gensim_sentences, window=5)
w2v_model.train(gensim_sentences,total_examples=len(gensim_sentences),epochs=10)
print("Word2vec vocab size: ",len(w2v_model.wv.index_to_key))
print("Most similar to girl: ",w2v_model.wv.most_similar(positive="girl"))
print("OK..")


w2v_model.wv.vectors[w2v_model.wv.key_to_index['girl']] #And index_to_key for dictionary in vocab class