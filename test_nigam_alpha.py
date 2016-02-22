import rfc822
import os
import csv
from numpy import random

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

from nbem import NaiveBayesEM


NEWSGROUPS_DIRECTORY = '20_newsgroups/'
FNAME_OUT = 'nigam_alpha_tests.csv'

# get this from Nigam -- they only use 10 but we could do more if we want
N_LABELED_SETS = 10

# note that group sizes should be interpreted relative to the ENTIRE set
TEST_GROUP_SIZE = 0.2
UNLABELED_GROUP_SIZE = 0.5

LABELED_SIZES = [15]

TEST_ALPHA_RESOLUTION = 10
TEST_ALPHA_MAX = 6

def generate_test_alphas(test_alpha_max, test_alpha_resolution):
    alphas_out = []

    for i in range(test_alpha_max * test_alpha_resolution):
        i_normed = float(i) / test_alpha_resolution
        alpha_i = pow(10, -i_normed)
        alphas_out.append(alpha_i)

    return alphas_out


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
                message = f.read().decode('ascii', errors='ignore')

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

def kfold_by_cat(labeled_docs, ndocs_per_class, shuffle=True):
    if shuffle:
        for cat in labeled_docs:
            random.shuffle(labeled_docs[cat])

    labeled_sets = []

    max_cat_size = max([len(v) for v in labeled_docs.values()])

    start_idx = 0
    while start_idx < max_cat_size:
        end_idx = start_idx + ndocs_per_class

        this_set = {}
        for cat in labeled_docs:
            this_set[cat] = labeled_docs[cat][start_idx:end_idx]

        labeled_sets.append(this_set)

        start_idx = end_idx

    return labeled_sets

def convert_docdict_to_array(d, vec, enc):
    docs_raw = []
    cats_raw = []

    for c in d:
        docs_raw += d[c]
        cats_raw += [c]*len(d[c])

    docs = vec.transform(docs_raw)
    cats = enc.transform(cats_raw)

    return docs, cats

def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


if __name__ == '__main__':
    all_docdict = load_documents(NEWSGROUPS_DIRECTORY)
    print sum([len(v) for v in all_docdict.values()])

    print "all read"

    ncats = len(all_docdict)

    ### build the data transformers
    # vectorizer for words
    # TODO: double check the tokenizer regex on this
    # NOTE: L1 norm to best reflect Nigam et al's
    #vec = TfidfVectorizer(use_idf=False, min_df=2, norm='l1', stop_words='english')
    vec = CountVectorizer(min_df=2, stop_words='english')
    docs_texts = []
    for c in all_docdict:
        for h,m in all_docdict[c]:
            #m = m.decode('ascii','ignore')
            docs_texts.append(m)
    vec.fit(docs_texts)
    del docs_texts

    # encoder for categories
    enc = LabelEncoder()
    enc.fit(all_docdict.keys())

    train_docdict, test_docdict = train_test_split_by_message_date(all_docdict)
    del all_docdict

    unlabeled_docdict, labeled_docdict = create_unlabeled_test_set(train_docdict)
    del train_docdict

    test_x, test_y = convert_docdict_to_array(test_docdict, vec, enc)
    del test_docdict
    unlabeled_x, unlabeled_y = convert_docdict_to_array(unlabeled_docdict, vec, enc)
    del unlabeled_docdict

    test_alphas = generate_test_alphas(TEST_ALPHA_MAX, TEST_ALPHA_RESOLUTION)
    #test_alphas = [0.002]


    with open(FNAME_OUT,'w') as fout:
        fwriter = csv.writer(fout)
        headerout = ['n.labeled','alpha','iteration','score']#,'with.em']
        fwriter.writerow(headerout)

        for labeled_size_experiment in LABELED_SIZES:
            print "testing size =", labeled_size_experiment

            experiment_scores_noem = []
            experiment_scores_em = []

            labeled_sets = kfold_by_cat(labeled_docdict, labeled_size_experiment)

            for alpha in test_alphas:
                i = 0
                total = len(labeled_sets)
                for labeled_set in labeled_sets[:N_LABELED_SETS]:
                    i += 1
                    labeled_x, labeled_y = convert_docdict_to_array(labeled_set, vec, enc)

                    rowout = [labeled_x.shape[0], alpha, i]

                    nb = MultinomialNB(alpha=alpha)
                    nb.fit(labeled_x, labeled_y)
                    noem_score = nb.score(test_x, test_y)
                    experiment_scores_noem.append(noem_score)
                    rowout.append(noem_score)

                    em = NaiveBayesEM(unlabeled_x, ncats, alpha=alpha, labeled_x=labeled_x, labeled_y=labeled_y)
                    em.models.append(nb)
                    em.runEM()

                    nb_out = em.models[-1]
                    em_score = nb_out.score(test_x, test_y)

                    rowout.append(em_score)

                    this_em_iter_scores = []
                    for model in em.models:
                        this_em_iter_scores.append(model.score(test_x, test_y))

                    print "size = %s, alpha = %s, score = %s, em_score = %s" % (total, alpha, noem_score, em_score)

                    fwriter.writerow(rowout)
