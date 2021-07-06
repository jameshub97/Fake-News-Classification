import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# work from https://www.kaggle.com/mohiteud/machine-learning-for-fake-news-detection
fake = pd.read_csv('../portfolio/neural_network/tests/Fake.csv', delimiter=',')
true = pd.read_csv('../portfolio/neural_network/tests/True.csv', delimiter=',')

fake['news_sentiment'] = 0
true['news_sentiment'] = 1
fake.drop(['date', 'subject'], axis=1, inplace=True)
true.drop(['date', 'subject'], axis=1, inplace=True)
dataset = pd.DataFrame()
dataset = true.append(fake)
dataset['text'] = dataset['title'] + dataset['text']
dataset.drop('title', axis=1, inplace=True)
print(dataset.size)
input_array = np.array(dataset['text'])
print(dataset.size)
print(dataset.head)
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# improve runtime
set(stopwords.words('english'))

corpus = []
for i in range(0, 40000):
    review = re.sub('[^a-zA-Z]', ' ', str(input_array[i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

print("loading!")
print(dataset.head)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:40000, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

print(dataset.head)

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
print("loading!")
classifier = GaussianNB()
print("loading!")
classifier.fit(X_train, y_train)
print("loading!")
classifierada = AdaBoostClassifier(base_estimator=classifier, learning_rate=1)
print("loading!")
classifierada.fit(X_train, y_train)
bagger = BaggingClassifier(base_estimator=classifier)
bagger.fit(X_train, y_train)
print("success!")

target_names = ['class 0', 'class1']
y_pred = classifier.predict(X_test)
y_predboost = classifierada.predict(X_test)
y_predbagger = bagger.predict(X_test)
#
from sklearn.metrics import confusion_matrix, classification_report
#
cm = confusion_matrix(y_test, y_pred)
cmnbboost = confusion_matrix(y_test, y_predboost)
cmbagger = confusion_matrix(y_test, y_predbagger)
print("NB","\n")
print(classification_report(y_test, y_pred,))
print("NB boosted","\n")
print(classification_report(y_test, y_predboost))
print("NB bagged","\n")
print(classification_report(y_test, y_predbagger, ))

from sklearn.linear_model import LogisticRegression

classifier1 = LogisticRegression(random_state=0)
classifier1.fit(X_train, y_train)
classifierada = AdaBoostClassifier(random_state=0, base_estimator=classifier1, learning_rate=1)
classifierada.fit(X_train, y_train)
bagger = BaggingClassifier(base_estimator=classifier1)
bagger.fit(X_train, y_train)
y_predL = classifier.predict(X_test)
y_predboost = classifierada.predict(X_test)
y_predbagger = bagger.predict(X_test)


cm1 = confusion_matrix(y_test, y_predL)
cm1boost = confusion_matrix(y_test, y_predL)
cmbagger= confusion_matrix(y_test, y_predbagger)
y_predbagger = bagger.predict(X_test)
print("LR","\n")
print(classification_report(y_test, y_pred, ))
print("LR boosted","\n")
print(classification_report(y_test, y_predboost, ))
print("LR bagged","\n")
print(classification_report(y_test, y_predbagger, ))


from sklearn.svm import SVC
classifier = SVC(probability=True, kernel='sigmoid')
# sigmoid 0.84 accuracy
classifier.fit(X_train, y_train)


y_predS = classifier.predict(X_test)

print("SVC","\n")
print(classification_report(y_test, y_predS))

cm5 = confusion_matrix(y_test, y_predS)

ax = plt.subplot()
sns.heatmap(cm5, annot=True, ax=ax, fmt='g');  # annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title("SVC");
ax.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax.yaxis.set_ticklabels(['Fake news', 'Real News']);
plt.show()

from sklearn.neighbors import KNeighborsClassifier

classifier4 = KNeighborsClassifier(n_neighbors=5, metric='cosine', p=2)
classifier4.fit(X_train, y_train)


model = abc.fit(X_train, y_train)

# Predicting the Test set results
y_predK = classifier4.predict(X_test)
y_predKboosted = abc.predict(X_test)
y_predKbagger = bagger.predict(X_test)

cm4 = confusion_matrix(y_test, y_predK)

print("Kneigh","\n")
print(classification_report(y_test, y_predK, ))



sns.heatmap(cm4, annot=True, ax=ax, fmt='g');  # annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title("KNeighbors Tree");
ax.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax.yaxis.set_ticklabels(['Fake news', 'Real News']);
plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.show()


