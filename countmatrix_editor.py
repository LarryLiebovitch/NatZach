import pandas as pd
import numpy as np
from scipy.sparse import hstack

def filter_countmatrix(keep_vocab, countmatrix):
    """
    Pass a table with a "col" column with col indices and "word"  column for corresponding vocab items, indicating vocabulary items to keep.
    Filters the countmatrix by dropping unnammed columns, and returns vocabulary with updated column numbers.
    """
    subvocab = keep_vocab.sort_values(by='col')
    keep_cols = subvocab['col'].values
    keep_mask =  pd.Series(range(countmatrix.shape[1])).isin(keep_cols).values
    subset = countmatrix[:,keep_mask]
    
    newvocab = subvocab.copy()
    newvocab.reset_index(drop=True,inplace=True)
    newvocab['col'] = subvocab.reset_index(drop=True).reset_index()['index']
    
    return newvocab, subset
    
    
def make_dicts(vocab_table):
    """
    Takes in vocab table with 'col' and 'word' columns indicating vocab item and column of count matrix.
    
    Returns word 2 code and code 2 word dictionaries.
    """
    
    w2c = {vocab_table.loc[k,'word']:vocab_table.loc[k,'col'] for k in range(vocab_table.shape[0])}
    c2w = {v:k for k,v in w2c.items()}
    return w2c,c2w

def quotient_countmatrix(eqreldict, vocabtable, countmatrix):
    """
    args: 
        eqrel: A dictionary whose keys are strings and values are lists of strings.
                The collection of sets consisting of keys together with the elements of their value lists, 
                together with the singletons of all vocab items not in the dictionary, should represent
                an equivalence relation on the vocabulary. The key should not be in the value list,
                and should stand for the representative of the equivalence class.
        
        vocabtable: Vocab table with 'col' and 'word' columns. The eqrel should represent an equivalence relation on the
                    words in the 'word' column of the vocab table
        
        countmatrix: A count matrix of a corpus indexed by the vocab table.
        
    returns:
                New vocab table and count matrix where words in the equivalence relation have been identified.
                The chosen representative will represent the equivalence class.
    """
    w2c = {vocabtable.loc[k,'word']:vocabtable.loc[k,'col'] for k in range(vocabtable.shape[0])}
    c2w = {v:k for k,v in w2c.items()}
    
    merged_cols = list()
    merged_vocab = list()
    droplist = list()
    
    for key, value in eqreldict.items():
        droplist.append(key)
        merged_vocab.append(key)
        newcol = countmatrix.getcol(w2c[key])
        
        values = eqreldict[key]
        for value in values:
            droplist.append(value)
            newcol += countmatrix.getcol(w2c[value])

        merged_cols.append(newcol)


    merged_cols = hstack(merged_cols)    
    nmerged = merged_cols.shape[1]
    merged_vocab = pd.Series(merged_vocab).reset_index()
    merged_vocab.columns = ['col', 'word']
    merged_vocab = merged_vocab.sort_values(by = 'col').reset_index(drop=True)
    
    keep_table = vocabtable[~vocabtable['word'].isin(droplist)].copy()
    subvocab, subset = filter_countmatrix(keep_table, countmatrix)
    subvocab = subvocab[['col','word']].copy()
    subvocab['col'] = subvocab['col'] + nmerged

    newvocab = pd.concat([merged_vocab,subvocab], axis = 0).reset_index(drop=True)
    newmatrix = hstack([merged_cols,subset])
    
    return newvocab, newmatrix
  
def w2c_totable(w2cdict):
    vocab = pd.DataFrame([(k,v) for k,v in w2cdict.items()])
    vocab.columns = ['word', 'col']
    vocab = vocab.sort_values(by = 'col').reset_index(drop=True)
    
    return vocab