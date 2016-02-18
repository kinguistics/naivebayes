import pickle
from nltk.corpus import brown
from sklearn.feature_extraction.text import CountVectorizer
from naivebayes import *
from nbem import NaiveBayesEM, count_docs_per_class, count_live_classes
from numpy import log, exp

def build_all_brown(subset_size=None):
    documents = []
    categories = []

    all_categories = set()

    try:
        fileids = brown.fileids()

        for fileid in fileids:
            if subset_size:
                if len(all_categories) > subset_size:
                    break
            category = brown.categories(fileid)[0]
            words = [x.lower() for x in brown.words(fileid)]

            documents.append(words)
            categories.append(category)

            all_categories.add(category)

        if subset_size != len(brown.categories()):
            # exclude the final item, since it's the sole member of the next group
            documents = documents[:-1]
            categories = categories[:-1]

        documents = [' '.join(d) for d in documents]

    except LookupError:
        ''' we don't have the Brown corpus via nltk on this machine '''
        try:
            with open('brown_docs_cats.pickle') as f:
                documents, categories = pickle.load(f)
        except IOError:
            raise Exception("can't load Brown Corpus via NLTK or file")

    #documents = [' '.join(d) for d in documents]

    '''
    # let's NOT get tempted to hide away the encoding
    # we'll probably need to access, e.g., the vectorizer, to do reverse
    # transformations once we want to interpret/evaluate the model

    doc_vectorizer = CountVectorizer()
    doc_vec = doc_vectorizer.fit_transform(documents)
    '''

    return documents, categories

'''
if __name__ == "__main__":
    NRUNS = 25

    docs, cats = build_all_brown(2)
    vectorizer = CountVectorizer()
    docs = vectorizer.fit_transform(docs)

    for ncats in range(2,21):
        for runnum in range(NRUNS):
            nbem = NaiveBayesEM(docs, ncats, fit_prior=False)
            nbem.runEM()

            nb = nbem.last_nb
            doc_classes = [v for v in count_docs_per_class(nb,docs) if v>0]
            priors = [v for v in exp(nb.class_log_prior_) if v > 0]
            live_classes = count_live_classes(nb,docs)
            likelihood = nbem.likelihood[-1]
            iterations = len(nbem.likelihood) - 1
            print '%s,%s,%s,%s,%s,"%s","%s"' % (ncats, runnum, iterations, live_classes, likelihood,doc_classes,priors)
'''
