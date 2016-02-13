from scipy import ndimage
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from brown_testing import *
from naivebayes import *
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    all_d,all_c = build_all_brown()
    
    vec = CountVectorizer()
    vec.fit(all_d)
    
    enc = LabelEncoder()
    enc.fit(all_c)
    
    # dictify the documents
    d_by_c = {}
    
    unique_ordered_labels = []
    
    for i in range(len(all_d)):
        this_doc = all_d[i]
        this_cat = all_c[i]
        
        if this_cat not in d_by_c:
            d_by_c[this_cat] = []
            unique_ordered_labels.append(this_cat)
        
        d_by_c[this_cat].append(this_doc)

    total_n_cats = len(unique_ordered_labels)
    all_alpha = np.array(range(100), dtype=float) / 100
    
    ncats_alpha_means = {}
    
    for ncat in range(2, total_n_cats+1):
        print ncat,"of",total_n_cats
        d_n = sum([d_by_c[c] for c in unique_ordered_labels[:ncat]])
        c_n = all_c[:len(d_n)]
        
        docs_n = vec.transform(d_n)
        cats_n = enc.transform(c_n)
        
        this_ncat_alpha_means = []
        for a in all_alpha:
            scores = cross_validate(docs_n, cats_n, alpha=a)
            this_ncat_alpha_means.append(np.mean(scores))
        
        ncats_alpha_means[ncat] = this_ncat_alpha_means
        plt.plot(all_alpha, this_ncat_alpha_means, label=ncat)

    plt.legend()
    plt.grid(True)
    
    plt.show()

alphameans_matrix = np.array(np.matrix(ncats_alpha_means.values()))
alphameans_gaus = ndimage.filters.gaussian_filter(alphameans_matrix, 2, mode='nearest')
alphameans_gaus.max()
fig, ax = plt.subplots()
heatmap = ax.pcolor(alphameans_gaus, cmap=plt.cm.Blues)
#ax.set_yticks(np.arange(alphameans_matrix.shape[1])+0.5, minor=False)
#ax.set_yticklabels(all_alpha, minor=False)
plt.show()
