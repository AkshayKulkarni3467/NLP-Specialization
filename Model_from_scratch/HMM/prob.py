import numpy as np
from tqdm import tqdm 



def word_given_tag(word, tag, train_bag):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    total_num_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_given_tag = len(w_given_tag_list)
    
    return (count_w_given_tag, total_num_tag)

def t2_given_t1(t2, t1, train_bag):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


        
def E_and_T_matrix(tokens,train_tagged_words):
    
    V = set(tokens) #vocabulary
    T = set([pair[1] for pair in train_tagged_words]) #Tags

    
    tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
    for i, t1 in tqdm(enumerate(list(T))):
        for j, t2 in enumerate(list(T)): 
            count_t2_t1,count_t1 = t2_given_t1(t2,t1,train_tagged_words)
            tags_matrix[i, j] = count_t2_t1/count_t1
    return tags_matrix