import glob
import csv
import pickle
from naivebayes import *

NRUNS = 25
CORPORA_BASEDIR = '/Users/etking/corpus/gender_convos'
ALL_CORPORA = ['bnc',
               'callhome',
               'colt',
               'fisher1',
               'fisher2',
               'santabarbara',
               'speeddate',
               'switchboard',
               'switchboardcell',
               'wellington']


def read_all_gender(corpora=None, byspeaker=False):
    if corpora is None:
        corpora = ALL_CORPORA

    docs = {}
    cats = {}

    for corpus in corpora:
        corpus_docs = []
        corpus_cats = []

        spkr_info_fname = '%s/%s.info' % (CORPORA_BASEDIR,corpus)
        with open(spkr_info_fname) as f:
            spkr_info = pickle.load(f)

        corpus_gender_count = len([key for key in spkr_info.keys() \
                                   if 'gender' in spkr_info[key]])
        corpus_size = len(spkr_info)
        corpus_gender_percent = float(corpus_gender_count) / corpus_size *100
        print "corpus %s: %s%% of %s have gender" % (corpus,
                                                     corpus_gender_percent,
                                                     corpus_size)

        for spkr in spkr_info:
            try: gender = spkr_info[spkr]['gender'].lower()
            except KeyError:
                continue

            if gender not in ['m','f']:
                continue

            spkr_doc_fnames = glob.glob('%s/%s/%s_*' % (CORPORA_BASEDIR,
                                                        corpus,
                                                        spkr))

            spkr_docs = []
            for spkr_doc_fname in spkr_doc_fnames:
                with open(spkr_doc_fname) as f:
                    s = f.read()
                    spkr_docs.append(s)

            # if each doc is a speaker, rather than a conversation
            if byspeaker:
                all_spkr_docs = ' '.join(spkr_docs)
                spkr_docs = [all_spkr_docs]

            spkr_cats = [gender] * len(spkr_docs)

            corpus_docs += spkr_docs
            corpus_cats += spkr_cats

        docs[corpus] = corpus_docs
        cats[corpus] = corpus_cats

    return docs, cats


if __name__ == '__main__':
    
    ### doing all the reading and model calculation
    '''
    print "reading all corpora"
    alldocs, allcats = read_all_gender(corpora=None)
    print "all corpora read"

    ### get the overall params
    docs = sum(alldocs.values())
    cats = sum(allcats.values())

    vectorizer = CountVectorizer()
    docs = vectorizer.fit_transform(docs)

    enc = LabelEncoder()
    cats = enc.fit_transform(cats)

    encoders = vectorizer, enc
    with open('gender_nb_encoders.pickle','w') as fout:
        pickle.dump(encoders, fout)

    nb = MultinomialNB()
    nb.fit(docs, cats)
    params = nb.class_log_prior_, nb.feature_log_prob_
    with open('gender_nb.pickle','w') as fout:
        pickle.dump(nb, fout)
    with open('gender_nb_params.pickle','w') as fout:
        pickle.dump(params, fout)

    
    ### now get all the crossval scores, using both the entire metacorpus and
    ###     each individual corpus
    all_scores = {}
    print "starting crossvals"

    all_scores['all_together'] = []
    for runnum in range(NRUNS):
        scores = cross_validate(docs, cats)
        all_scores['all_together'].append(scores)


    for corpus in ALL_CORPORA:
        docs = alldocs[corpus]
        cats = allcats[corpus]

        vectorizer = CountVectorizer()
        docs = vectorizer.fit_transform(docs)

        enc = LabelEncoder()
        cats = enc.fit_transform(cats)
        
        encoders = vectorizer, enc
        with open('gender_nb_%s_encoders.pickle' % corpus,'w') as fout:
            pickle.dump(encoders, fout)
        
        # overall model for fratio stuff later
        nb = MultinomialNB()
        nb.fit(docs, cats)
        with open('gender_%s_nb.pickle' % corpus,'w') as fout:
            pickle.dump(nb, fout)

        all_scores[corpus] = []
        for runnum in range(NRUNS):
            print "crossvals",corpus,runnum
            scores = cross_validate(docs, cats)
            all_scores[corpus].append(scores)


    with open('gender_crossval_scores.pickle','w') as fout:
        pickle.dump(all_scores, fout)


    ### for docs that have bad genders
    # gooddocs = docs[[i for i in range(cats.shape[0]) if cats[i] not in enc.transform(['u','na'])]]
    '''
    
    ### analyzing the saved models from above
    fratios = {}
    for corpus in ALL_CORPORA:
        fratios[corpus] = {}
        with open('gender_nb_%s_encoders.pickle' % corpus) as f:
            vectorizer, enc = pickle.load(f)
        with open('gender_%s_nb.pickle' % corpus) as f:
            nb = pickle.load(f)
        
        f_class = enc.transform('f')
        fratio = get_fratio(nb, f_class)
        
        rev_vocab = make_reverse_vocabulary(vectorizer)
        for i in range(len(fratio)):
            val = fratio[i]
            word = rev_vocab[i]
            fratios[corpus][word] = val
    
    # make sure to get the "everything" corpus
    corpus = "all_together"
    fratios[corpus] = {}

    with open('gender_nb_encoders.pickle') as f:
        vectorizer, enc = pickle.load(f)
    with open('gender_nb.pickle') as f:
        nb = pickle.load(f)
    
    f_class = enc.transform('f')
    fratio = get_fratio(nb, f_class)
    
    rev_vocab = make_reverse_vocabulary(vectorizer)
    for i in range(len(fratio)):
        val = fratio[i]
        word = rev_vocab[i]
        fratios[corpus][word] = val
    
    corpus_names = ['all_together']+ALL_CORPORA
    header = ['word'] + corpus_names
    with open('all_corpus_fratios.csv','w') as fout:
        fwriter = csv.writer(fout)
        fwriter.writerow(header)
        
        all_words = fratios['all_together'].keys()
        for word in all_words:
            rowout = [word]
            for corpus in corpus_names:
                try: corpus_word_fratio = fratios[corpus][word]
                except KeyError: corpus_word_fratio = 0
                
                rowout.append(corpus_word_fratio)
            
            fwriter.writerow(rowout)