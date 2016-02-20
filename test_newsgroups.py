from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import sklearn.cross_validation as cv

import csv

newsgroups = fetch_20newsgroups(subset='all', remove=('headers','footers'))

min_freq_results = {}

with open('newsgroups_min_df_tests_train50_tfidf.csv','w') as fout:
    fwriter = csv.writer(fout)
    header = ['min.freq','iteration','score']
    fwriter.writerow(header)
    
    for min_freq in range(1,21):
        min_freq_results[min_freq] = []
        
        for i in range(10):
            print min_freq, i
            
            vec = TfidfVectorizer(use_idf=False, min_df=min_freq)
            docs = vec.fit_transform(newsgroups.data)
            
            cats = newsgroups.target
            
            nb = MultinomialNB(alpha=0.2)
            x_train, x_test, y_train, y_test = cv.train_test_split(docs, cats, train_size=0.5)
            nb.fit(x_train, y_train)
            
            predicted = nb.predict(x_test)
            confusion = confusion_matrix(y_test, predicted)
            score = float(confusion.diagonal().sum()) / confusion.sum()
            
            rowout = [min_freq, i, score]
            fwriter.writerow(rowout)
            
            min_freq_results[min_freq].append(confusion)
            