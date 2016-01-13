import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
from numpy import log, exp, isnan, isinf, ceil, sum


def build_all_brown(subset=False):
    documents = []
    categories = []

    all_categories = set()

    for fileid in brown.fileids():
        if subset:
            if len(all_categories) > 2:
                break
        category = fileid[:2]
        words = [x.lower() for x in brown.words(fileid)]

        documents.append(words)
        categories.append(category)

        all_categories.add(category)

    if subset:
        # exclude the final item, since it's the sole member of the third group
        return documents[:-1], categories[:-1]
    else:
        return documents, categories

def build_crossval_indices(n,k):
    indices = [int(ceil((float(n)/k)*i)) for i in range(k+1)]
    return indices

def get_crossval_split(l, indices, i):
    test_start_idx = indices[i]
    test_end_idx = indices[i+1]

    train_l = l[:test_start_idx] + l[test_end_idx:]
    test_l = l[test_start_idx:test_end_idx]

    return train_l, test_l

def shuffle_paired_lists(l1, l2):
    zipped = zip(l1, l2)
    random.shuffle(zipped)

    unzipped1 = [v[0] for v in zipped]
    unzipped2 = [v[1] for v in zipped]

    return unzipped1, unzipped2

if __name__ == '__main__':
    nfolds = 10
    
    docs, cats = build_all_brown(True)
    docs = [' '.join(d) for d in docs]
    
    doc_vectorizer = CountVectorizer(docs)
    doc_vectorizer.fit(docs)
    
    nb = MultinomialNB()
    
    fold_indices = build_crossval_indices(len(docs), nfolds)
    docs, cats = shuffle_paired_lists(docs, cats)
    
    sk_nb_accuracies = []
    for crossval_iter in range(10):
        for fold_number in range(nfolds):
            train_docs, test_docs = get_crossval_split(docs, fold_indices, fold_number)
            train_vector = doc_vectorizer.transform(train_docs)
            test_vector = doc_vectorizer.transform(test_docs)
            
            train_cats, test_cats = get_crossval_split(cats, fold_indices, fold_number)
            
            
            nb.fit(train_vector, train_cats)
            acc = nb.score(test_vector, test_cats)
            print fold_number, acc
            sk_nb_accuracies.append(acc)
            '''
            fold_accurate_classification_count = 0
            for test_idx in range(len(test_docs)):
                test_doc = test_docs[test_idx]
                test_cat = test_cats[test_idx]
        
                predicted_category = self.classify(test_doc)
        
                doc_number = self.documents.index(test_doc)
                print "document #%s should be %s; classified as %s" % (doc_number, argmax_dict(test_cat), predicted_category)
                if predicted_category == argmax_dict(test_cat):
                    fold_accurate_classification_count += 1
        
            fold_accuracy = float(fold_accurate_classification_count) / len(test_docs)
            print "fold:",fold_number+1, "... accuracy:",fold_accuracy
            '''