import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import rand
from nltk.corpus import brown
from numpy import log, exp, isnan, isinf, ceil, sum
from sklearn.utils.extmath import logsumexp

class NaiveBayesEM(object):
    def __init__(self, documents, n_categories, randomize=False, max_iterations=20):
        self.documents = documents
        self.vocab = self.generate_vocab()

        self.categories = range(n_categories)

        self.p_categories = []
        self.p_words_by_category = []

        self.randomize = randomize

        self.max_iterations = max_iterations
        self.likelihood = []
        self.all_nb = []

    def runEM(self):
        self.initializeEM()

        for iter_n in range(self.max_iterations):
            done = False
            
            try: prev_likelihood = self.likelihood[-1]
            except IndexError: prev_likelihood = None

            nb = MultinomialNB(self.documents, None)
            nb.categories = self.categories
            self.all_nb.append(nb)

            self.e_step(nb)
            self.m_step(nb)
            #del nb

            ### CHECK LIKELIHOOD CHANGE
            this_likelihood = nb.calculate_likelihood()
            self.likelihood.append(this_likelihood)
            if this_likelihood == prev_likelihood:                
                done = True

            print iter_n, this_likelihood, nb.count_classifications()
            if done:
                break

    def initializeEM(self):
        p_categories = {}
        for c in self.categories:
            randfloat = random.random()
            if self.randomize:
                p_categories[c] = randfloat
            else:
                # make it mostly equal
                p_categories[c] = float(1+(0.05*randfloat)) / len(self.categories)
        p_categories = scale_dictionary(p_categories)

        p_words_by_category = {}
        for c in self.categories:
            cwords = {}
            for w in self.vocab:
                randfloat = random.random()
                if self.randomize:
                    cwords[w] = random.random()
                else:
                    wordcount = self.vocab[w]
                    cwords[w] = float(wordcount+(0.05*randfloat)) / len(self.vocab)
            p_words_by_category[c] = scale_dictionary(cwords)

        self.p_categories.append(p_categories)
        self.p_words_by_category.append(p_words_by_category)

    def e_step(self, nb):
        nb.p_categories = self.p_categories[-1]
        nb.p_words_by_category = self.p_words_by_category[-1]

        nb.doc_categories = []

        for doc in self.documents:
            doc_class = nb.soft_classify(doc)
            #nb.likelihood += sum(doc_class.values())
            nb.doc_categories.append(doc_class)

    def m_step(self, nb):
        nb.train()
        self.p_categories.append(nb.p_categories)
        self.p_words_by_category.append(nb.p_words_by_category)

    def generate_vocab(self):
        vocab = {}

        for doc in self.documents:
            for word in doc:
                vocab[word] = doc[word]
                
        return vocab
    
    def calculate_category_words(self, n=10):
        nb = self.all_nb[-1]
        predictions = nb.count_classifications()
        
        baseline = self.vocab
        
        good_cats = {}
        
        for c in predictions:
            c_worddict = nb.p_words_by_category[c]
            diff_from_baseline = [(w, c_worddict[w] - baseline[w]) for w in baseline]
            diff_from_baseline.sort(cmp=lambda x,y: cmp(x[1],y[1]))
            
            good_cats[c] = diff_from_baseline[:n] + diff_from_baseline[-n:]
        
        return good_cats
            
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

if __name__ == '__main__':
    skdocs, skcats = build_all_brown(True)
    skdocs = [' '.join(d) for d in skdocs]
    
    doc_vectorizer = CountVectorizer(skdocs)
    doc_vectorizer.fit(skdocs)
    
    sknb = MultinomialNB()
    doc_vec = doc_vectorizer.transform(skdocs)
    sknb.fit(doc_vec, skcats)
