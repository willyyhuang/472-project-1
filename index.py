import numpy as np
import os
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from pandas import *

datadir="BBC"
categories=["tech", "entertainment", "politics", "business", "sport"]
txt_array=[];
all_array=[];
for category in categories:  # for each class
    path = os.path.join(datadir,category)  # create path to class
    for txt in os.listdir(path):  # iterate over each text file per class
        txt_array.append(txt)
    all_array.append(len(txt_array))
    txt_array=[]
fig = plt.figure()
plt.bar(categories,all_array)
fig.savefig('BBC-distribution.pdf', dpi=fig.dpi)

corpus=load_files(container_path=datadir,encoding="latin1")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus.data)

X_train, X_test, y_train, y_test = train_test_split(X,corpus.target, test_size=0.2,random_state=None)

clf = MultinomialNB()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

f = open('bbc-performance.txt', 'w')
f.write('---------------------------------------------------------------------------------\n')
f.write('0 - business category\n')
f.write('1 - entertainment category\n')
f.write('2 - politics category\n')
f.write('3 - sport category\n')
f.write('4 - tech category\n')
f.write('---------------------------------------------------------------------------------\n')
f.write('b) Confusion matrix\n')
matrix = confusion_matrix(y_test, predicted)
f.write(DataFrame(matrix).to_string() + '\n')
f.write('---------------------------------------------------------------------------------\n')
f.write('c) Classification Report\n\n')
f.write(classification_report(y_test, predicted, target_names=corpus.target_names) + '\n')
f.write('d) Accuracy, Macro Average F1, Weighted Average\n')
f.write('accuracy: ' + str(accuracy_score(y_test, predicted)) + '\n')
f.write('macro average f1: ' + str(f1_score(y_test, predicted, average='macro')) + '\n')
f.write('weighted average: ' + str(f1_score(y_test, predicted, average='weighted')) + '\n')
f.write('---------------------------------------------------------------------------------\n')
f.write('e) Prior Probabilities\n')
f.write('---------------------------------------------------------------------------------\n')
f.write('f) Size of vocabulary\n')
f.write('size of vocabulary ' + str(len((vectorizer.vocabulary_))))
f.write('---------------------------------------------------------------------------------\n')





