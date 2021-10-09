import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

classInstances = ['business', 'entertainment', 'politics', 'sport', 'tech']
# step 2
plt.bar(classInstances, [501, 386, 417, 511, 401])
plt.savefig('BBC-distribution.pdf')

# step 3, 4
corpus = datasets.load_files('./BBC', encoding='latin1')
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus.data)

# step 5
X_train, X_test, y_train, y_test = train_test_split(X,corpus.target, test_size=0.2,random_state=None)

# step 6
clf = MultinomialNB()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

# step 7
f = open('bbc-performance.txt', 'w')
