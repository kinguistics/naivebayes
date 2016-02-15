from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from brown_testing import *
from naivebayes import *

from sklearn.utils.extmath import logsumexp


from numpy import where, min, max, mean, average, median
import numpy as np

import sklearn.cross_validation as cv

import csv
import pandas

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

def optimize_alpha(x_train, x_dev, y_train, y_dev, min_a=0, max_a=10):
    asize = ALPHA_OPTMZN_RESOLUTION
    all_alpha = array(range(min_a*asize, max_a*asize)).astype('float') / asize
    #all_alpha = array(range(asize)).astype('float') / asize

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

def train_dev_test_split(x, y, sizes=np.array([0.5, 0.25, 0.25])):
    sizes = sizes / sum(sizes)
    train_size = sizes[0]
    dev_size = sizes[1] / (1 - train_size)

    x_train, x_nontrain, y_train, y_nontrain = \
            cv.train_test_split(x, y, train_size=train_size)
    x_dev, x_test, y_dev, y_test = \
            cv.train_test_split(x_nontrain, y_nontrain, train_size=dev_size)

    return x_train, x_dev, x_test, y_train, y_dev, y_test

def log_likelihood(nb, docs):
    jll = nb._joint_log_likelihood(docs)
    ll_by_class = logsumexp(jll,axis=1)
    ll = sum(ll_by_class)

    return ll

if __name__ == '__main__':
    all_d, all_c = build_all_brown()

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
    all_loglikes = {}
    all_mean_extremes = {}
    all_median_extremes = {}


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
            x_train, x_dev, x_test, y_train, y_dev, y_test = \
                    train_dev_test_split(low_freq_removed, cats)
            #best_alpha = optimize_alpha(x_train, x_dev, y_train, y_dev)
            best_alpha = 0.2

            nb = MultinomialNB(alpha=best_alpha)
            nb.fit(x_train, y_train)
            score = nb.score(x_test, y_test)


            all_scores[hl_pair] = mean(score)

            all_sizes[hl_pair] = size

            ### TODO: want to cv this as well???
            ###     figure out the summary stat
            ll = log_likelihood(nb, low_freq_removed)
            all_loglikes[hl_pair] = ll

            n_most_extreme = find_n_characteristic_indices(nb, low_freq_removed, how='odds_ratio')
            mean_nme = mean([len(v) for v in n_most_extreme.values()])
            all_mean_extremes[hl_pair] = mean_nme
            median_nme = median([len(v) for v in n_most_extreme.values()])
            all_median_extremes[hl_pair] = median_nme

            print hl_pair, size, minmax, score, mean_nme, median_nme, ll


            #all_alphas[hl_pair] = best_alpha

    with open('hl_freq_tests.csv','w') as fout:
        fwriter = csv.writer(fout)
        #header = ['high_idx','low_idx','lowest_freq','highest_freq','alpha','ncols','score']
        header = ['high_idx','low_idx','lowest_freq','highest_freq','ncols','score','mean_n_tied_best','median_n_tied_best','loglikelihood']
        fwriter.writerow(header)

        hlpairs = all_scores.keys()

        for pair in sorted(hlpairs):
            h_idx,l_idx = pair
            h_freq,l_freq = all_minmax[pair]
            #alpha = all_alphas[pair]
            ncols = all_sizes[pair]
            score = all_scores[pair]
            mean_nme = all_mean_extremes[hl_pair]
            median_nme = all_median_extremes[hl_pair]
            loglike = all_loglikes[hl_pair]

            rowout = [h_idx, l_idx, h_freq, l_freq, ncols, score, mean_nme, median_nme, loglike]
            fwriter.writerow(rowout)

    '''
    HOLD ON TO THIS FOR IPYTHON BUT DON'T DO IN NON-INTERACTIVE MODE

    df = pandas.DataFrame.from_csv('hl_freq_tests.csv', index_col=None)
    max_hcut = max(df['high_idx'])
    max_lcut = max(df['low_idx'])
    hl_scores = np.zeros((max_hcut+1, max_lcut+1))

    for row_idx in range(len(df['score'])):
        h_idx = df['high_idx'][row_idx]
        l_idx = df['low_idx'][row_idx]
        score = df['score'][row_idx]
        hl_scores[h_idx][l_idx] = score

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.plot(df['lowest_freq'], df['score'],'o')
    plt.show()
    #heatmap = ax.pcolor(hl_scores, cmap=plt.cm.Blues)
    #ax.set_yticks(np.arange(alphameans_matrix.shape[1])+0.5, minor=False)
    #ax.set_yticklabels(all_alpha, minor=False)
    '''
