import random
import csv
import numpy as np
from numpy import ceil, floor, histogram2d, sum, array, log
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from brown_testing import build_all_brown

### MODEL EVALUATION
# functions for cross-validation of MultinomialNB
def cross_validate(docs, cats, nfolds=10, **kwargs):
    nsamples = len(cats)
    cats_types = list(set(cats))
    shuffled_indices = range(nsamples)
    random.shuffle(shuffled_indices)

    scores = []

    start_idx = 0
    foldsize = float(nsamples) / (nfolds)

    for k in range(nfolds):
        stop_idx = int(round((k+1) * foldsize))

        train_indices = shuffled_indices[:start_idx]+shuffled_indices[stop_idx:]
        test_indices = shuffled_indices[start_idx:stop_idx]

        train_docs, train_cats = docs[train_indices], cats[train_indices]
        test_docs, test_cats = docs[test_indices], cats[test_indices]

        nb = MultinomialNB(**kwargs)
        nb.partial_fit(train_docs, train_cats, classes=cats_types)
        start_idx = stop_idx

        score = nb.score(test_docs, test_cats)
        scores.append(score)

    return scores

def build_confusion_matrix(nb, test_docs, test_cats):
    ''' THIS IS BROKEN IN A WAY I CAN'T FIGURE OUT

        should get confusion matrix m such that
        sum(m.diagonal()) / sum(m)
        == nb.score(test_docs, test_cats)

        but for some reason these two are not always equal'''
    class_size = len(nb.classes_)
    predicted = nb.predict(test_docs)
    #return test_cats,predicted,class_size
    m, xe, ye = histogram2d(test_cats, predicted, bins=((class_size,class_size)))

    return m

### MODEL INTERPRETATION
# functions for odds-ratio of fitted model
def find_n_most_extreme_odds_ratios(nb, word_log_prob, n=10, numerator_row=0):
    odds_ratio = get_odds_ratio(nb, word_log_prob)

    n_most_extreme = {}

    nclasses = len(nb.classes_)
    for class_idx in range(nclasses):
        class_odds = odds_ratio[class_idx]
        odds_with_indices = list(enumerate(class_odds))
        odds_with_indices.sort(cmp=lambda x,y: cmp(x[1],y[1]))

        n_highest_indices = [o[0] for o in odds_with_indices[-n:]]
        n_lowest_indices = [o[0] for o in odds_with_indices[:n]]

        n_most_extreme[class_idx] = (n_highest_indices, n_lowest_indices)

    return n_most_extreme


def get_odds_ratio(nb, word_log_probs):
    nclasses = len(nb.classes_)
    overall_log_prob_matrix = word_log_probs.repeat(nclasses,axis=0)

    return nb.feature_log_prob_ - overall_log_prob_matrix

def find_n_most_extreme_fratios(nb, n=10, numerator_row=0):
    fratio = get_fratio(nb, numerator_row)

    class_odds = fratio
    odds_with_indices = list(enumerate(class_odds))
    odds_with_indices.sort(cmp=lambda x,y: cmp(x[1],y[1]))

    n_highest_indices = [o[0] for o in odds_with_indices[-n:]]
    n_lowest_indices = [o[0] for o in odds_with_indices[:n]]

    return n_highest_indices, n_lowest_indices


def get_fratio(nb, numerator_row=0):
    nclasses = len(nb.classes_)
    denominator_row = [i for i in range(nclasses) if i!=numerator_row][0]

    return nb.feature_log_prob_[numerator_row] - nb.feature_log_prob_[denominator_row]

def construct_word_log_prob(docs):
    docs_count = array(docs.sum(axis=0))
    docs_total = np.ones(docs_count.shape) * docs.sum()

    docs_log_prob = log(docs_count) - log(docs_total)

    return docs_log_prob
