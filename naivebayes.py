import random
import pickle
from nltk.corpus import brown
from numpy import log, exp, isnan, isinf, ceil, sum

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

def argmax_dict(d):
    # check to make sure there *is* a single highest value
    if (len(d) > 1) and (len(set(d.values())) == 1):
        return None

    d_list = d.items()
    d_list.sort(cmp=lambda x,y: cmp(x[1],y[1]), reverse=True)

    return d_list[0][0]

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
        
        self.vocab = set(sum([d.keys() for d in documents]))

        self.p_categories = p_categories
        self.p_words_by_category = p_words_by_category

    def classify(self, document):
        """ returns list of most probable classes """
        p_by_category = self.soft_classify(document)

        best_category = argmax_dict(p_by_category)

        return best_category

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
                p_by_category[category] += p_word_by_category * document[word]

        #scaled_p_by_category = scale_log_dictionary(p_by_category)

        #p_by_category = scale_log_dictionary(p_by_category)

        return p_by_category
    
    def count_classifications(self):
        classes = {}
        for doc in self.documents:
            predicted = self.classify(doc)
            if predicted not in classes:
                classes[predicted] = 0
            classes[predicted] += 1
        return classes
            

    def train(self, documents=None, categories=None):
        if documents is None:
            documents = self.documents
        if categories is None:
            categories = self.doc_categories

        # count dicts
        c_categories = self.softcount_categories(categories)
        c_words_by_category = self.softcount_words_by_category(documents, categories)
        # add-one smoothing
        c_words_by_category = self.smooth_word_counts_by_category(c_words_by_category)

        # scale the count dicts to get probabilities
        p_categories = scale_log_dictionary(c_categories)

        p_words_by_category = {}
        for category in c_words_by_category:
            p_words_by_category[category] = scale_log_dictionary(c_words_by_category[category])

        # set the object variables
        self.p_categories = p_categories
        self.p_words_by_category = p_words_by_category

    def softcount_categories(self, soft_classifications=None):
        if soft_classifications is None:
            soft_classifications = self.doc_categories
        
        soft_classifications = soft_classifications

        c_categories = {}

        for doc_softclass in soft_classifications:
            doc_softclass = scale_log_dictionary(doc_softclass)
            for category in doc_softclass:
                # add (logged) softcounts of categories
                if category not in c_categories:
                    c_categories[category] = log(0)
                c_categories[category] = logAdd(c_categories[category],
                                                doc_softclass[category])

        return c_categories

    def softcount_words_by_category(self, documents=None, soft_classifications=None):
        if documents is None:
            documents = self.documents
        if soft_classifications is None:
            soft_classifications = self.doc_categories

        assert len(documents) == len(soft_classifications)

        c_words_by_category = {}

        for doc_idx in range(len(documents)):
            document = documents[doc_idx]
            category_softclass = scale_log_dictionary(soft_classifications[doc_idx])

            for category in category_softclass:
                p_doc_category = category_softclass[category]

                if category not in c_words_by_category:
                    c_words_by_category[category] = {}

                for word in document:
                    #c_words_by_category[category][word] = document[word] * p_doc_category
                    for wcount in range(document[word]):
                    #log_wordcount = document[word]
                        
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
        lexicon = self.vocab
        '''
        for category in c_words_by_category:
            lexicon = lexicon.union(set(c_words_by_category[category]))
        '''

        smoothed_words_by_category = {}

        for category in c_words_by_category:
            smoothed_words_by_category[category] = {}
            for word in lexicon:
                try: true_count = c_words_by_category[category][word]
                except KeyError: true_count = 0

                smoothed_words_by_category[category][word] = logAdd(true_count, log(1))

        return smoothed_words_by_category

    def crossval(self, nfolds=10):
        fold_indices = build_crossval_indices(len(self.documents), nfolds)
        docs, cats = shuffle_paired_lists(self.documents, self.doc_categories)

        accuracies = []

        for fold_number in range(nfolds):
            train_docs, test_docs = get_crossval_split(docs, fold_indices, fold_number)
            train_cats, test_cats = get_crossval_split(cats, fold_indices, fold_number)

            self.train(documents=train_docs, categories=train_cats)

            fold_accurate_classification_count = 0
            for test_idx in range(len(test_docs)):
                test_doc = test_docs[test_idx]
                test_cat = test_cats[test_idx]

                predicted_category = self.classify(test_doc)

                #print "document #%s should be %s; classified as %s" % (doc_number, argmax_dict(test_cat), predicted_category)
                if predicted_category == argmax_dict(test_cat):
                    fold_accurate_classification_count += 1

            fold_accuracy = float(fold_accurate_classification_count) / len(test_docs)                
            accuracies.append(fold_accuracy)
            print "fold:",fold_number+1, "... accuracy:",fold_accuracy
            if fold_accuracy < 0.5:
                return train_docs, test_docs, train_cats, test_cats


        #return accuracies


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

            nb = NaiveBayes(self.documents, None)
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

def convert_categories_to_probs(catlist):
    problist = []

    for category in catlist:
        d = {category: 0}
        problist.append(d)

    return problist

def doc2dict(doc):
    d = {}
    for word in doc:
        if word not in d:
            d[word] = 0
        d[word] += 1
    return d

if __name__ == '__main__':
    try:
        docs, cats = build_all_brown(subset=True)
    except:
        with open('brown_docs_cats.pickle') as f:
            docs, cats = pickle.load(f)
    docs = [doc2dict(doc) for doc in docs]
    catprobs = convert_categories_to_probs(cats)

    '''
    # manual folding
    cb_start = cats.index('cb')
    five_percent_idx = len(docs)/20
    test_start = cb_start - five_percent_idx
    test_end = cb_start+(len(docs)/20)

    docs_test = docs[test_start:test_end]
    cats_test = catprobs[test_start:test_end]
    docs_train = docs[:test_start] + docs[test_end:]
    cats_train = catprobs[:test_start] + catprobs[test_end:]
    
    #binary_docs = [list(set(d)) for d in docs]
    '''
    nb = NaiveBayes(docs, catprobs)

    accs = nb.crossval()

    '''
    nbem = NaiveBayesEM(docs, 15, randomize=False)
    nbem.runEM()
    print nbem.calculate_category_words()
    '''