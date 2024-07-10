from dataloader import dataloader
from nltk.tokenize import word_tokenize
from viterbi import Viterbi

_,_,train_tagged_words,tokens = dataloader()


sentence_test = 'Hello, I am Akshay and I am currently studying NLP. It is a branch of articial intelligence and has various research opportunities in it.'
words = word_tokenize(sentence_test)

tagged_seq = Viterbi(words = words,tokens=tokens,train_tagged_words=train_tagged_words)

words = [tup[0] for tup in tagged_seq]
POS = [tup[1] for tup in tagged_seq]


print("The words tokenized are {}".format(words))
print('The POS tags for them are {}'.format(POS))