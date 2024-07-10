from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize

def dataloader():
    pos_tagged = list(nltk.corpus.treebank.tagged_sents())
    train_set, test_set = train_test_split(pos_tagged,test_size=0.3)
    train_tagged_words = [tup for sent in train_set for tup in sent]
    tokens = [pair[0] for pair in train_tagged_words]
    return train_set,test_set,train_tagged_words,tokens