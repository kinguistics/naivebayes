import random
import csv
import numpy as np
from numpy import array, log
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from brown_testing import *

### MODEL EVALUATION

### MODEL INTERPRETATION
def find_characteristic_class_words(nb, docs, vectorizer, class_n, n_words=10, how='log_ratio'):
    indices = find_n_characteristic_indices(nb, docs, n_words, how)
    class_indices = indices[class_n]
    
    # SHOULD be sorted
    class_indices.reverse()
    
    revvoc = make_reverse_vocabulary(vectorizer)
    words = [(revvoc[n], metric) for (n,metric) in class_indices]
    return words

# functions for odds-ratio of fitted model
def find_n_characteristic_indices(nb, docs, n=10, how='log_ratio'):
    if how == 'odds_ratio':
        word_log_prob = construct_word_log_prob(docs)
        words_metric = get_odds_ratio(nb, word_log_prob)
    else:
        words_metric = get_log_ratio(nb)

    n_most_extreme = {}

    nclasses = len(nb.classes_)
    for class_idx in range(nclasses):
        class_metrics = words_metric[class_idx]
        metrics_with_indices = list(enumerate(class_metrics))
        metrics_with_indices.sort(cmp=lambda x,y: cmp(x[1],y[1]))

        highest_val_in_list = metrics_with_indices[-1][1]
        i = len(metrics_with_indices)-1
        while metrics_with_indices[i][1] == highest_val_in_list:
            i -=1
        n_highest_indices = metrics_with_indices[i:]
        '''
        n_lowest_indices = [o for o in metrics_with_indices[:n]]

        n_most_extreme[class_idx] = (n_highest_indices, n_lowest_indices)
        '''
        n_most_extreme[class_idx] = n_highest_indices

    return n_most_extreme


def get_odds_ratio(nb, word_log_probs):
    nclasses = len(nb.classes_)
    overall_log_prob_matrix = word_log_probs.repeat(nclasses,axis=0)

    return nb.feature_log_prob_ - overall_log_prob_matrix


def get_log_ratio(nb, numerator_row=0):
    nclasses = len(nb.classes_)
    denominator_row = [i for i in range(nclasses) if i!=numerator_row][0]

    return nb.feature_log_prob_[numerator_row] - nb.feature_log_prob_[denominator_row]


def construct_word_log_prob(docs):
    docs_count = array(docs.sum(axis=0))
    docs_total = np.ones(docs_count.shape) * docs.sum()

    docs_log_prob = log(docs_count) - log(docs_total)

    return docs_log_prob


def make_reverse_vocabulary(vectorizer):
    revvoc = {}
        
    vocab = vectorizer.vocabulary_
    for w in vocab:
        i = vocab[w]
        
        revvoc[i] = w
    
    return revvoc