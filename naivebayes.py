import random
from nltk.corpus import brown
from numpy import log, exp, isnan, isinf

def logAdd(logX, logY):
    # make logX the max of the wo
    if logY > logX:
        logX, logY = logY, logX

    if isnan(logY) or isinf(logY):
        return logX

    negDiff = logY - logX
    #print negDiff
    if negDiff < -20:
        return logX

    return (logX + log(1.0 + exp(negDiff)))

def logSum(log_sequence):
    return reduce(lambda x,y: logAdd(x,y), log_sequence)

def scale_dictionary(d):
    """ dict(str : int) -> dict(str : float) """
    scaled = {}

    total = log(sum(d.values()))
    for key in d:
        scaled[key] = log(d[key]) - total

    return scaled

def scale_log_dictionary(d):
    scaled = {}

    total = logSum(d.values())
    for key in d:
        scaled[key] = d[key] - total

    return scaled


class NaiveBayes(object):
    """
    A NaiveBayes object trains or applies a Naive Bayes model of text.

    :param documents: the texts to classify
    :type documents: list(list(str))
    :param doc_categories: a list of dicts of the same length as documents, with
                       the probability of document assignment to each category
    :type doc_categories: list(dict(category: float) len(categories) == len(documents)
    :param p_categories: the category probabilities
    :type p_categories: dict(c in categories : float), or None
    :param p_words_by_categories: P(word | category)
    :type p_words_by_categories: dict(int : dict(str : float)), or
                                 dict(str : dict(str : float)). or
                                 None
    """
    def __init__(self,
                 documents,
                 doc_categories=None,
                 p_categories=None,
                 p_words_by_category=None):
        self.documents = documents

        self.doc_categories = doc_categories
        self.categories = set()
        if self.doc_categories is not None:
            for doc_cat in doc_categories:
                self.categories = self.categories.union(set(doc_cat.keys()))
        elif p_categories is not None:
            self.categories = p_categories.keys()
        self.categories = list(self.categories)
        self.categories.sort()

        self.p_categories = p_categories
        self.p_words_by_category = p_words_by_category

        self.likelihood = log(1)

    def classify(self, document):
        """ returns list of most probable classes """
        p_by_category = self.soft_classify(document)

        p_list = p_by_category.items()
        p_list.sort(cmp=lambda x,y: cmp(x[1],y[1]), reverse=True)

        return p_list[0][0]

    def soft_classify(self, document):
        if (self.p_words_by_category is None):
            raise BaseException, "model is not trained"

        p_by_category = {}
        for category in self.categories:
            p_by_category[category] = self.p_categories[category]

        for word in document:
            for category in self.categories:
                try: p_word_by_category = self.p_words_by_category[category][word]
                except KeyError:
                    p_word_by_category = log(0)
                p_by_category[category] += p_word_by_category

        #scaled_p_by_category = scale_log_dictionary(p_by_category)

        #p_by_category = scale_log_dictionary(p_by_category)

        return p_by_category

    def train(self):
        # count dicts
        c_categories = self.softcount_categories()
        c_words_by_category = self.softcount_words_by_category()

        # add-one smoothing
        #smoothed_word_counts = self.smooth_word_counts_by_category(c_words_by_category)
        # scale the count dicts to get probabilities
        p_categories = scale_log_dictionary(c_categories)

        p_words_by_category = {}
        for category in c_words_by_category:
            p_words_by_category[category] = scale_log_dictionary(c_words_by_category[category])

        # set the object variables
        self.p_categories = p_categories
        self.p_words_by_category = p_words_by_category


    def count_categories(self):
        c_categories = {}

        for category in self.categories:
            # keep counts of categories
            if category not in c_categories:
                c_categories[category] = 0
            c_categories[category] += 1

        return c_categories

    def softcount_categories(self, soft_classifications=None):
        if soft_classifications is None:
            soft_classifications = self.doc_categories

        c_categories = {}

        for doc_softclass in soft_classifications:
            for category in doc_softclass:
                # add (logged) softcounts of categories
                if category not in c_categories:
                    c_categories[category] = log(0)
                c_categories[category] = logAdd(c_categories[category],
                                                doc_softclass[category])

        return c_categories

    def count_words_by_category(self):
        c_words_by_category = {}

        # count words by category
        for doc_idx in range(len(self.documents)):
            document = self.documents[doc_idx]
            category = self.categories[doc_idx]

            if category not in c_words_by_category:
                c_words_by_category[category] = {}

            for word in document:
                if word not in c_words_by_category[category]:
                    c_words_by_category[category][word] = 0
                c_words_by_category[category][word] += 1

        return c_words_by_category

    def softcount_words_by_category(self, soft_classifications=None):
        if soft_classifications is None:
            soft_classifications = self.doc_categories

        c_words_by_category = {}

        for doc_idx in range(len(self.documents)):
            document = self.documents[doc_idx]
            category_softclass = soft_classifications[doc_idx]

            for category in category_softclass:
                p_doc_category = category_softclass[category]

                if category not in c_words_by_category:
                    c_words_by_category[category] = {}

                for word in document:
                    if word not in c_words_by_category[category]:
                        c_words_by_category[category][word] = log(0)
                    c_words_by_category[category][word] = logAdd(c_words_by_category[category][word],
                                                          p_doc_category)

        return c_words_by_category

    def calculate_likelihood(self):
        likelihood = log(1)
        for doc_class in self.doc_categories:
            #likelihood += sum(doc_class.values())
            likelihood += max(doc_class.values())
        return likelihood

    def smooth_word_counts_by_category(self,c_words_by_category):
        """ smoothing fucks up the classification for some reason """
        lexicon = set()
        for category in c_words_by_category:
            lexicon = lexicon.union(set(c_words_by_category[category]))

        smoothed_words_by_category = {}

        for category in c_words_by_category:
            smoothed_words_by_category[category] = {}
            for word in lexicon:
                try: true_count = c_words_by_category[category][word]
                except KeyError: true_count = 0

                smoothed_words_by_category[category][word] = true_count + 1

        return smoothed_words_by_category

class NaiveBayesEM(object):
    def __init__(self, documents, n_categories, max_iterations=10):
        self.documents = documents
        self.vocab = self.generate_vocab()

        self.categories = range(n_categories)

        self.p_categories = []
        self.p_words_by_category = []

        self.max_iterations = max_iterations
        self.likelihood = []
        self.all_nb = []

    def runEM(self):
        self.initializeEM()

        for iter_n in range(self.max_iterations):
            try: prev_likelihood = self.likelihood[-1]
            except IndexError: prev_likelihood = None

            nb = NaiveBayes(self.documents, None)
            nb.categories = self.categories
            self.all_nb.append(nb)

            self.e_step(nb)
            self.m_step(nb)
            #del nb

            ### CHECK LIKELIHOOD CHANGE
            this_likelihood = nb.calculate_likelihood()
            self.likelihood.append(this_likelihood)

            print iter_n, this_likelihood

    def initializeEM(self):
        p_categories = {}
        for c in self.categories:
            p_categories[c] = random.random()
            #p_categories[c] = float(1) / len(self.categories)
        p_categories = scale_dictionary(p_categories)

        p_words_by_category = {}
        for c in self.categories:
            cwords = {}
            for w in self.vocab:
                cwords[w] = random.random()
                #cwords[w] = float(1) / len(self.vocab)
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
        vocab = set()

        for doc in self.documents:
            docset = set(doc)
            vocab = vocab.union(docset)

        vocab = list(vocab)
        vocab.sort()

        return vocab

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

    return documents, categories

def convert_categories_to_probs(catlist):
    problist = []

    for category in catlist:
        d = {category: 0}
        problist.append(d)

    return problist

if __name__ == '__main__':
    docs, cats = build_all_brown(subset=True)
    binary_docs = [list(set(d)) for d in docs]

    nb = NaiveBayes(docs, convert_categories_to_probs(cats))
    nb.train()

    '''
    nbem = NaiveBayesEM(docs[:-1], 2)
    nbem.runEM()
    '''
