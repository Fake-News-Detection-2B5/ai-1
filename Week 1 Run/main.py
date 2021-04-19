import nltk
import string
import pandas as pd
import seaborn as sns
from sklearn import svm
from nltk import tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

data = pd.read_csv("Sample.csv", delimiter='\t', encoding='utf-8')

# Data Preparation

# Removing title and id

data.drop(["title"], axis=1, inplace=True)
data.drop(["public_id"], axis=1, inplace=True)

# Converting to lowercase

data['text'] = data['text'].apply(lambda x: x.lower())

# Removing punctuation signs


def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


data['text'] = data['text'].apply(punctuation_removal)

# Removing Stopwords (words filtered out before natural language processing)

stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))


# Removing dashes and underscores

def line_removal(text):
    all_list = [char for char in text if char not in "-–—_"]
    clean_str = ''.join(all_list)
    return clean_str


data['text'] = data['text'].apply(line_removal)
# print(data)

# Data Exploration

data.groupby(['our rating'])['text'].count().plot(kind="bar")
# plt.show()

# Word Cloud View (Fake)

fake_data = data[data["our rating"] == "FALSE"]
all_words = ' '.join([text for text in fake_data.text])

word_cloud = WordCloud(width=1000, height=1000, max_font_size=110, collocations=False).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
# plt.show()

# Word Cloud View (True)

true_data = data[data["our rating"] == "TRUE"]
all_words = ' '.join([text for text in true_data.text])

word_cloud = WordCloud(width=1000, height=1000, max_font_size=110, collocations=False).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
# plt.show()

# Word Cloud View (Partially False)

partially_false_data = data[data["our rating"] == "partially false"]
all_words = ' '.join([text for text in partially_false_data.text])

word_cloud = WordCloud(width=1000, height=1000, max_font_size=110, collocations=False).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
# plt.show()

# Separating by White Spaces

token_space = tokenize.WhitespaceTokenizer()

# Most Frequent Words


def counter(text, column_text, quantity):
    words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()), "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns="Frequency", n=quantity)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_frequency, x="Word", y="Frequency", color='blue')
    ax.set(ylabel="Count")
    plt.xticks(rotation='vertical')
    plt.show()

# Fake News


counter(data[data["our rating"] == "FALSE"], "text", 20)

# True News


counter(data[data["our rating"] == "TRUE"], "text", 20)

# Partially Fake


counter(data[data["our rating"] == "partially false"], "text", 20)

# Split Data

x_train, x_test, y_train, y_test = train_test_split(data['text'], data['our rating'], test_size=0.2, random_state=42)

# Decision Tree

decision_tree_pipe = Pipeline([('vector', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('model', DecisionTreeClassifier
                               (criterion='entropy', max_depth=20, splitter='best', random_state=42))])
decision_tree_model = decision_tree_pipe.fit(x_train, y_train)
decision_tree_prediction = decision_tree_model.predict(x_test)

print("Decision Tree")
print(confusion_matrix(y_test, decision_tree_prediction))
print(classification_report(y_test, decision_tree_prediction, zero_division=0))


# K-Nearest Neighbors

k_neighbors_pipe = Pipeline([('vector', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('model', KNeighborsClassifier())])
k_neighbors_model = k_neighbors_pipe.fit(x_train, y_train)
k_neighbors_prediction = k_neighbors_model.predict(x_test)

print("K-Nearest Neighbors")
print(confusion_matrix(y_test, k_neighbors_prediction))
print(classification_report(y_test, k_neighbors_prediction, zero_division=0))


# Support Vector Machine

support_vector_machine_pipe = Pipeline([('vector', CountVectorizer()),
                                        ('tfidf', TfidfTransformer()),
                                        ('model', svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))])
support_vector_machine_model = support_vector_machine_pipe.fit(x_train, y_train)
support_vector_machine_prediction = support_vector_machine_model.predict(x_test)

print("Support Vector Machine")
print(confusion_matrix(y_test, support_vector_machine_prediction))
print(classification_report(y_test, support_vector_machine_prediction, zero_division=0))


# Naive-Bayes

naive_bayes_pipe = Pipeline([('vector', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('model', MultinomialNB())])
naive_bayes_model = naive_bayes_pipe.fit(x_train, y_train)
naive_bayes_prediction = naive_bayes_model.predict(x_test)

print("Naive-Bayes")
print(confusion_matrix(y_test, naive_bayes_prediction))
print(classification_report(y_test, naive_bayes_prediction, zero_division=0))
