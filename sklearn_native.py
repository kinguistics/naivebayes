import random
import pickle
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
from numpy import log, exp, isnan, isinf, ceil, sum, resize
from sklearn.utils.extmath import logsumexp

NRUNS = 100

def count_docs_per_class(nb):
    return nb.predict_proba(doc_vec).sum(axis=0)

def count_live_classes(nb):
    docs_per_class = count_docs_per_class(nb)
    return len(docs_per_class.nonzero()[0])

class NaiveBayesEM(object):
    def __init__(self, documents, n_categories, max_iterations=20, randomize=False):
        self.documents = documents
        self.n_categories = n_categories
        
        self.n_samples, self.n_features = self.documents.shape
        
        self.class_log_prior_shape = (self.n_categories,)
        self.feature_log_prob_shape = (self.n_categories, self.n_features)
        
        self.class_log_priors = []
        self.feature_log_probs = []

        self.randomize = randomize

        # when/how to stop the EM iterations
        self.max_iterations = max_iterations
        self.likelihood = []

    def runEM(self):
        self.initializeEM()

        for iter_n in range(self.max_iterations):
            done = False
            
            try: prev_likelihood = self.likelihood[-1]
            except IndexError: prev_likelihood = None

            nb = MultinomialNB(self.documents, None)
            # add faked "classes_" attribute to force it to think it's been trained
            nb.classes_ = np.ndarray((self.n_categories,))
            # and add the random parameters to actually "train" it
            nb.class_log_prior_ = self.class_log_priors[-1]
            nb.feature_log_prob_ = self.feature_log_probs[-1]

            soft_predictions = self.e_step(nb)
            nb = self.m_step(soft_predictions)
            
            self.last_nb = nb
            
            ### CHECK LIKELIHOOD CHANGE
            jll = nb._joint_log_likelihood(self.documents)
            best_likelihoods = jll.max(axis=1)
            this_likelihood = sum(best_likelihoods)
            
            self.likelihood.append(this_likelihood)
            if this_likelihood == prev_likelihood:                
                done = True
                pass

            #print iter_n, this_likelihood #, nb.count_classifications()
            #print iter_n, this_likelihood, count_live_classes(nb)
            if done:
                break

    def initializeEM(self):
        if self.randomize:
            class_log_prior_ = generate_normed_rand_log_prob(self.class_log_prior_shape)
            feature_log_prob_ = generate_normed_rand_log_prob(self.feature_log_prob_shape)
        else:
            uniform_class_counts = np.ones(self.class_log_prior_shape)
            class_log_prior_ = generate_normed_rand_log_prob(self.class_log_prior_shape,
                                                             count_vec=uniform_class_counts,
                                                             max_alpha=0)
            
            doc_vec_counts = resize(self.documents.sum(0), self.feature_log_prob_shape)
            feature_log_prob_ = generate_normed_rand_log_prob(self.feature_log_prob_shape, 
                                                              count_vec=doc_vec_counts, 
                                                              max_alpha=10)
        
        self.class_log_priors.append(class_log_prior_)
        self.feature_log_probs.append(feature_log_prob_)

    def e_step(self, nb):
        nb.class_log_prior_ = self.class_log_priors[-1]
        nb.feature_log_prob_ = self.feature_log_probs[-1]

        soft_predictions = nb.predict_proba(self.documents)
        return soft_predictions

    def m_step(self, soft_predictions):
        nb = MultinomialNB()
        
        for category_idx in range(self.n_categories):
            catvec = np.zeros(self.n_samples)
            catvec += category_idx
            
            cat_weights = soft_predictions.T[category_idx]
            nb.partial_fit(self.documents, 
                           catvec, 
                           classes=range(self.n_categories),
                           sample_weight=cat_weights)
            
        self.class_log_priors.append(nb.class_log_prior_)
        self.feature_log_probs.append(nb.feature_log_prob_)
        
        return nb
            
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

def generate_normed_rand_log_prob(vecshape, count_vec=None, max_alpha=10):
    if count_vec is None:
        rand_prob = np.random.random(vecshape)
    else:
        rand_to_add = ceil(np.random.random(vecshape) * max_alpha)
        rand_prob = count_vec + rand_to_add
    rand_log_prob = log(rand_prob)
    
    norm_axis = vecshape.index(max(vecshape))
    log_norm = logsumexp(rand_log_prob, norm_axis)
    log_norm_vec = resize(log_norm, vecshape)
    
    normed_rand_log_prob = rand_log_prob - log_norm_vec
    
    return normed_rand_log_prob

if __name__ == '__main__':
    try:
        skdocs, skcats = build_all_brown(subset=True)
    except:
        with open('brown_docs_cats.pickle') as f:
            skdocs, skcats = pickle.load(f)
    skdocs = [' '.join(d) for d in skdocs]
    
    doc_vectorizer = CountVectorizer(skdocs)
    doc_vectorizer.fit(skdocs)
    doc_vec = doc_vectorizer.transform(skdocs)

    print "run.n, n.iterations,n.classes,likelihood"
    # simulate the likelihood landscape
    for run_number in range(NRUNS):
        nbem = NaiveBayesEM(doc_vec, 15, max_iterations=50)
        nbem.runEM()
        
        nb = nbem.last_nb
        live_classes = count_live_classes(nb)
        likelihood = nbem.likelihood[-1]
        iterations = len(nbem.likelihood) - 1
        
        with open('all_brown_em_tests/run_%s.pickle' % run_number,'w') as fout:
            pickle.dump(nb, fout)
        
        print "%s,%s,%s,%s" % (run_number, iterations, live_classes, likelihood)