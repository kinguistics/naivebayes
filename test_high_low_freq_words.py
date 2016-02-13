from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from brown_testing import *

from numpy import where, min, max, mean, average

import sklearn.cross_validation as cv

import csv

MAXN_HIGH = None
MAXN_LOW = None
ALPHA_OPTMZN_RESOLUTION = 100

def remove_one_frequency(docs, fn=min):
    counts_m = docs.sum(axis=0)
    
    # this is a pain but necessary for where()
    counts = array(counts_m.tolist()[0])

    # max or min, presumably
    fn_counts = fn(counts)
    
    indices_to_remove = where(counts == fn_counts)[0]
    indices_to_keep = array([i for i in range(docs.shape[1]) if i not in indices_to_remove])
    
    # COLUMNS
    return docs.transpose()[indices_to_keep].transpose()

def optimize_alpha(x_train, x_dev, y_train, y_dev):
    asize = ALPHA_OPTMZN_RESOLUTION
    all_alpha = array(range(asize)).astype('float') / asize
    
    all_scores = []
    for alpha in all_alpha:
        nb = MultinomialNB(alpha=alpha)
        nb.fit(x_train, y_train)
        score = nb.score(x_dev, y_dev)
        all_scores.append(score)
    
    #max_score_idx = all_scores.index(max(all_scores))
    #return all_alpha, all_scores
    #return all_alpha[max_score_idx]
    
    # let's try a weighted mean
    return average(all_alpha, weights=all_scores)

if __name__ == '__main__':
    all_d, all_c = build_all_brown(3)
    
    vec = CountVectorizer()
    docs = vec.fit_transform(all_d)
    
    nwords = docs.shape[1]
    
    enc = LabelEncoder()
    cats = enc.fit_transform(all_c)
    
    nb = MultinomialNB()

    all_scores = {}
    all_sizes = {}
    all_minmax = {}
    all_alphas = {}

    high_freq_removed = docs #.transpose()[array(range(100))].transpose()

    if MAXN_HIGH is None:
        n_high = nwords
    if MAXN_LOW is None:
        n_low = nwords

    for high_freq_removed_n in range(n_high):
        
        if high_freq_removed_n > 0:
            high_freq_removed = remove_one_frequency(high_freq_removed, max)
        
        high_size = high_freq_removed.shape[1]
        if high_size == 0:
            break
        
        low_freq_removed = high_freq_removed
        
        for low_freq_removed_n in range(n_low):
            hl_pair = (high_freq_removed_n, low_freq_removed_n)
            
            if low_freq_removed_n > 0:
                low_freq_removed = remove_one_frequency(low_freq_removed, min)
            
            # we're done here if we've emptied the docs
            size = low_freq_removed.shape[1]
          
            if size == 0:
                break
            
            # ready to test
            '''
            score = cv.cross_val_score(nb, low_freq_removed, cats, cv=10)
            '''
            
            counts = low_freq_removed.sum(axis=0)
            minmax = (counts.min(), counts.max())
            all_minmax[hl_pair] = minmax
            
            # need to optimize alpha for this size
            x_train, x_nontrain, y_train, y_nontrain = \
                    cv.train_test_split(low_freq_removed, cats, train_size=0.5)
            x_dev, x_test, y_dev, y_test = \
                    cv.train_test_split(x_nontrain, y_nontrain, train_size=0.5)
            best_alpha = optimize_alpha(x_train, x_dev, y_train, y_dev)
            
            nb = MultinomialNB(alpha=best_alpha)
            nb.fit(x_train, y_train)
            score = nb.score(x_test, y_test)

            print hl_pair, size, minmax, score

            all_scores[hl_pair] = mean(score)

            all_sizes[hl_pair] = size
            
            
            all_alphas[hl_pair] = best_alpha

    with open('hl_freq_tests.csv','w') as fout:
        fwriter = csv.writer(fout)
        header = ['high_idx','low_idx','lowest_freq','highest_freq','alpha','ncols','score']
        fwriter.writerow(header)
        
        hlpairs = all_scores.keys()
        
        for pair in sorted(hlpairs):
            h_idx,l_idx = pair
            h_freq,l_freq = all_minmax[pair]
            alpha = all_alphas[pair]
            ncols = all_sizes[pair]
            score = all_scores[pair]
            
            rowout = [h_idx, l_idx, h_freq, l_freq, alpha, ncols, score]
            fwriter.writerow(rowout)