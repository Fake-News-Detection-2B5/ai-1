import warnings
import numpy as np
import pandas as pd
from sklearn import svm
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(500)

data = pd.read_csv("Sample.csv", delimiter='\t', encoding='utf-8')

# Data Preparation

# Removing title and id

data.drop(["title"], axis=1, inplace=True)
data.drop(["public_id"], axis=1, inplace=True)

# Converting to lowercase

data['text'] = [entry.lower() for entry in data['text']]

# Tokenization: Each tweet will be remembered as a set of words


data['text'] = [word_tokenize(entry) for entry in data['text']]

# Removal of Stop Words, Non-Alphanumerical Characters and Lemmatization

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['ADJ'] = wn.ADJ
tag_map['VERB'] = wn.VERB
tag_map['ADV'] = wn.ADV

for index, entry in enumerate(data['text']):
    final_words = []
    word_lemmatized = WordNetLemmatizer()

    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            final_word = word_lemmatized.lemmatize(word, tag_map[tag[0]])
            final_words.append(final_word)

    # The processed sets will be stored in final_text

    data.loc[index, 'final_text'] = str(final_words)

data.drop(["text"], axis=1, inplace=True)

# print(data["final_text"])

# Split Data

x_train, x_test, y_train, y_test = train_test_split(data["final_text"], data["our rating"], test_size=0.1)

# Encoding from String to Float

Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

# Word Vectorization

tfidf_vector = TfidfVectorizer(max_features=5000)
tfidf_vector.fit(data['final_text'])

x_train_tfidf = tfidf_vector.transform(x_train)
x_test_tfidf = tfidf_vector.transform(x_test)

print(tfidf_vector.vocabulary_)
print(x_train_tfidf)


# Testing Decision Tree on Encoding

DT = DecisionTreeClassifier(criterion='entropy', max_depth=20, splitter='best', random_state=42)
DT.fit(x_train_tfidf, y_train)
decision_tree_prediction = DT.predict(x_test_tfidf)

print("Decision Tree")
print(confusion_matrix(y_test, decision_tree_prediction))
print(classification_report(y_test, decision_tree_prediction, zero_division=0))


# Testing K-Nearest Neighbors on Encoding

KNN = KNeighborsClassifier()
KNN.fit(x_train_tfidf, y_train)
k_neighbors_prediction = KNN.predict(x_test_tfidf)

print("K-Nearest Neighbors")
print(confusion_matrix(y_test, k_neighbors_prediction))
print(classification_report(y_test, k_neighbors_prediction, zero_division=0))


# Testing Support Vector Machine on Encoding

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(x_train_tfidf, y_train)
svm_prediction = SVM.predict(x_test_tfidf)

print("Support Vector Machine")
warnings.filterwarnings("ignore")
print(confusion_matrix(y_test, svm_prediction))
print(classification_report(y_test, svm_prediction))


# Testing Naive-Bayes on Encoding

NB = MultinomialNB()
NB.fit(x_train_tfidf, y_train)
naive_bayes_prediction = NB.predict(x_test_tfidf)

print("Naive-Bayes")
print(confusion_matrix(y_test, naive_bayes_prediction))
print(classification_report(y_test, naive_bayes_prediction, zero_division=0))
