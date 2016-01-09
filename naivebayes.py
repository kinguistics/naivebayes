from nltk.corpus import brown
from numpy import log, exp, isnan

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
    :param categories: the category each text belongs to
    :type categories: list(int) or list(str). len(categories) == len(documents)
    :param p_categories: the category probabilities
    :type p_categories: dict(c in categories : float), or None
    :param p_words_by_categories: P(word | category)
    :type p_words_by_categories: dict(int : dict(str : float)), or
                                 dict(str : dict(str : float)). or
                                 None
    """
    def __init__(self,
                 documents,
                 categories=None,
                 p_categories=None,
                 p_words_by_category=None):
        self.documents = documents
        self.categories = categories
        self.p_categories = p_categories
        self.p_words_by_category = p_words_by_category

    def classify(self, document):
        """ returns list of most probable classes """
        p_by_category = self.soft_classify(document)

        p_list = p_by_category.items()
        p_list.sort(cmp=lambda x,y: cmp(x[1],y[1]), reverse=True)
        
        '''
        ### from here down will return all equiprobable best categories
        ### honestly I'm not sure what is gained from this
        best_score = p_list[0][1]
        
        category_idx = 1
        while category_idx < len(p_list):
            if p_list[category_idx][1] != best_score:
                break
            category_idx += 1

        return p_list[:category_idx]
        '''
        
        return p_list[0][0]

    def soft_classify(self, document):
        if (self.p_categories is None) or (self.p_words_by_category is None):
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

        return p_by_category

    def train(self):
        # count dicts
        c_categories = self.count_categories()
        c_words_by_category = self.count_words_by_category()

        # add-one smoothing
        #smoothed_word_counts = self.smooth_word_counts_by_category(c_words_by_category)
        # scale the count dicts to get probabilities
        p_categories = scale_dictionary(c_categories)

        p_words_by_category = {}
        for category in c_words_by_category:
            p_words_by_category[category] = scale_dictionary(c_words_by_category[category])

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
    
    def softcount_categories(self, soft_classifications):
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
    
    def softcount_words_by_category(self, soft_classifications):
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
                    c_words_by_category = logAdd(c_words_by_category[category][word],
                                                 p_doc_category)
        
        return c_words_by_category

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

if __name__ == '__main__':
    docs, cats = build_all_brown(subset=True)
    
    nb = NaiveBayes(docs, cats)