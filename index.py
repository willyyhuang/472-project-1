import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer

# step 2
plt.plot(['business', 'entertainment', 'politics', 'sport', 'tech'], [501, 386, 417, 511, 401])
plt.savefig('BBC-distribution.pdf')

# step 3, 4
corpus = datasets.load_files('./data/BBC-20210914T194535Z-001/BBC', encoding='latin1')
vectorizer = CountVectorizer()
term_document_matrix = vectorizer.fit_transform(corpus.data)

# step 5


