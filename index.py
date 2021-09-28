import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

classInstances = ['business', 'entertainment', 'politics', 'sport', 'tech']
# step 2
plt.bar(classInstances, [501, 386, 417, 511, 401])
plt.savefig('BBC-distribution.pdf')

# step 3, 4
corpus = datasets.load_files('./data/BBC-20210914T194535Z-001/BBC', encoding='latin1')
vectorizer = CountVectorizer()
term_document_matrix = vectorizer.fit_transform(corpus.data)

# step 5
training_set, testing_set = train_test_split(corpus.data, test_size=0.2, random_state=None)

# step 6
clf = MultinomialNB()
# clf.fit(training_set, classInstances)
# print(clf.predict(testing_set))