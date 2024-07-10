import pandas as pd
from tqdm import tqdm 
from prob import word_given_tag
from prob import E_and_T_matrix




def Viterbi(words,tokens, train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_tagged_words]))
    
    tags_matrix = E_and_T_matrix(tokens,train_tagged_words)
    
    tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))
    
    for key, word in tqdm(enumerate(words)):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                
            # compute emission and state probabilities
            prob,count = word_given_tag(words[key],tag,train_bag=train_tagged_words)
            emission_p = prob/count
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))