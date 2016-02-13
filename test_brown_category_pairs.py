from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from brown_testing import *
from naivebayes import *
from itertools import combinations
import numpy as np

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
    total_combinations = total_n_cats * (total_n_cats - 1) / 2
    
    all_pairs = combinations(unique_ordered_labels, 2)
    pair_n = 1
    
    accuracy_by_combo = {}
    
    for cname1,cname2 in all_pairs:
        print float(pair_n) / total_combinations
        pair_n += 1
        
        d1 = d_by_c[cname1]
        d2 = d_by_c[cname2]
        
        d = d1 + d2
        docs = vec.transform(d)
        
        c1 = [cname1] * len(d1)
        c2 = [cname2] * len(d2)
        
        c = c1 + c2
        cats = enc.transform(c)
        
        scores = cross_validate(docs, cats, alpha=0.1)
        
        cpair = (cname1, cname2)
        accuracy_by_combo[cpair] = np.mean(scores)
        
        ## TODO: plot this