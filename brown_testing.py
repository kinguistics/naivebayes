import pickle
from nltk.corpus import brown
from sklearn.feature_extraction.text import CountVectorizer
from naivebayes import NaiveBayesEM, count_docs_per_class, count_live_classes
from numpy import log, exp

def build_all_brown(subset=False):
    documents = []
    categories = []

    all_categories = set()

    try:
        fileids = brown.fileids()

        for fileid in fileids:
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
            documents = documents[:-1]
            categories = categories[:-1]

    except LookupError:
        ''' we don't have the Brown corpus via nltk on this machine '''
        try:
            with open('brown_docs_cats.pickle') as f:
                documents, categories = pickle.load(f)
        except IOError:
            raise Exception("can't load Brown Corpus via NLTK or file")

    documents = [' '.join(d) for d in documents]
    doc_vectorizer = CountVectorizer()
    doc_vec = doc_vectorizer.fit_transform(documents)

    return doc_vec, categories

if __name__ == "__main__":
    NRUNS = 25

    docs, cats = build_all_brown(subset=False)

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
