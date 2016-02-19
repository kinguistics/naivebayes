import rfc822
import os
from numpy import random

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from nbem import NaiveBayesEM


NEWSGROUPS_DIRECTORY = '20_newsgroups/'

# note that group sizes should be interpreted relative to the ENTIRE set
TEST_GROUP_SIZE = 0.2
UNLABELED_GROUP_SIZE = 0.5

def load_documents(dirname):
    dir_walker = os.walk(dirname)
    
    all_documents = {}
    
    for directory in dir_walker:
        dirpath, dirnames, filenames = directory
        
        if not len(filenames):
            continue
        
        category = dirpath.split(os.sep)[-1]
        if not len(category):
            continue
            
        if category not in all_documents:
            all_documents[category] = []
        
        for fname in filenames:
            full_fname = os.sep.join([dirpath, fname])
            
            with open(full_fname) as f:
                headers = rfc822.Message(f)
                message = f.read()
            
            all_documents[category].append((headers, message))
    
    return all_documents

def train_test_split_by_message_date(all_documents):
    train_docs = {}
    test_docs = {}
    
    for category in all_documents:
        category_docs = all_documents[category]
        
        ndocs = len(category_docs)
        split_idx = ndocs - int(ndocs * TEST_GROUP_SIZE)
        
        category_docs_dated = [(rfc822.parsedate(h['date']), d) \
                               for h,d in category_docs]
        category_docs_dated.sort(cmp=lambda x,y: cmp(x[0],y[0]))

        cat_train_dated = category_docs_dated[:split_idx]
        cat_train = [v[1] for v in cat_train_dated]
        train_docs[category] = cat_train
        
        cat_test_dated = category_docs_dated[split_idx:]
        cat_test = [v[1] for v in cat_test_dated]
        test_docs[category] = cat_test
            
    return train_docs, test_docs

def create_unlabeled_test_set(train_docs):
    unlabeled_percentage = UNLABELED_GROUP_SIZE / (1 - TEST_GROUP_SIZE)
    
    unlabeled_docs = {}
    labeled_docs = {}
    
    for cat in train_docs:        
        unlabeled_docs[cat] = []
        labeled_docs[cat] = []

        cat_docs = train_docs[cat]
        
        ndocs = len(cat_docs)
        n_unlabeled = int(unlabeled_percentage * ndocs)
        
        unlabeled_indices = set(random.choice(range(ndocs), n_unlabeled, replace=False))
        
        for i in range(ndocs):
            if i in unlabeled_indices:
                unlabeled_docs[cat].append(cat_docs[i])
            else:
                labeled_docs[cat].append(cat_docs[i])
            
    return unlabeled_docs, labeled_docs

def kfold_by_cat(labeled_docs, ndocs_per_class):
    labeled_sets = []
        
    

if __name__ == '__main__':
    all_d = load_documents(NEWSGROUPS_DIRECTORY)
    
    ### build the data transformers
    # vectorizer for words
    # TODO: double check the tokenizer regex on this
    # TODO: put in normalization here???
    vec = CountVectorizer()
    docs_texts = []
    for c in all_d:
        for h,m in all_d[c]:
            docs_texts.append(m)
    vec.fit(m)
    
    # encoder for categories
    enc = LabelEncoder()
    enc.fit(all_d.keys())
    
    train_docs, test_docs = train_test_split_by_message_date(all_d)
    unlabeled_docs, labeled_docs = create_unlabeled_test_set(train_docs)
    