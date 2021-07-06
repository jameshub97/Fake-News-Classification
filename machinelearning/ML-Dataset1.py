import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import numpy as np

import seaborn as sns
from sklearn.model_selection import train_test_split

# Loading data
fake = pd.read_csv('files/fnn_testfake.csv', delimiter=',', encoding='unicode_escape', low_memory=False)
true = pd.read_csv('files/fnn_testreal.csv', delimiter=',', encoding='unicode_escape', low_memory=False)

import random

# adding fake and true columns
fake['sentiment'] = 0
true['sentiment'] = 1
# combining datasets
dataset = pd.DataFrame()
dataset = true.append(fake)

print(len(dataset))
import random

print("success")
import matplotlib.pyplot as plt
origial_size = dataset.size
half_size = int(dataset.size / 2)

# dropping non textual data
column = ['news_url', 'id', 'tweet_ids']
dataset = dataset.drop(columns=column)
input_array = np.array(dataset['title'])
print(dataset.head)

from scipy import stats

corpus = []
# removal of stop words and stemming, then added to corpus
for i in range(0, 1056):
    review = re.sub('[^a-zA-Z]', ' ', str(input_array[i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances

# Different n-gram paramaters are assigned
cv = CountVectorizer(max_features=1056, ngram_range=(1, 1))
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:1056, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print("Set : Train set has total {0} entries with {1:.2f}% fake news, {2:.2f}% real news".format(len(X_train),
                                                                                                 (len(X_train[
                                                                                                          y_train == 0]) / (
                                                                                                          len(
                                                                                                              X_train) * 1.)) * 100,
                                                                                                 (len(X_train[
                                                                                                          y_train == 1]) / (
                                                                                                          len(
                                                                                                              X_train) * 1.)) * 100))

import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2

tfidf = CountVectorizer(max_features=30000, ngram_range=(1, 1))
X_tfidf = tfidf.fit_transform(dataset.title)
# Chi2 Graph
y = dataset.sentiment
chi2score = chi2(X_tfidf, y)[0]
plt.figure(figsize=(16, 8))
scores = list(zip(tfidf.get_feature_names(), chi2score))
chi2 = sorted(scores, key=lambda x: x[1])
topchi2 = list(zip(*chi2[-40:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x, topchi2[1], align='center', alpha=0.5)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
plt.show();
import timeit
import time

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

start = time.time()

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

end = time.time()

startboosted = time.time()

from sklearn.naive_bayes import GaussianNB
classifierada = AdaBoostClassifier(base_estimator=classifier, learning_rate=1)
classifierada.fit(X_train, y_train)
y_predboost = classifierada.predict(X_test)

endboosted = time.time()

startbagged = time.time()
bagger = BaggingClassifier(base_estimator=classifier)
bagger.fit(X_train, y_train)
y_predbagger = bagger.predict(X_test)
endbagged = time.time()
from sklearn.metrics import confusion_matrix, classification_report

cmNB = confusion_matrix(y_test, y_pred)
cmnNBboost = confusion_matrix(y_test, y_predboost)
cmNBbag = confusion_matrix(y_test, y_predbagger)
import datetime

import matplotlib.pyplot as plt

import datetime
import numpy as np

print("NB", "\n")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
nbms = round(time.time() - start, 2)
print("NB boosted", "\n")
print(classification_report(y_test, y_predboost, target_names=['negative', 'positive']))
boostedms = round(time.time() - startboosted, 2)
print("NB bagged", "\n")
print(classification_report(y_test, y_predbagger, target_names=['negative', 'positive']))

baggedms = round(time.time() - startbagged, 2)

print(nbms)
import matplotlib.ticker as ticker

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

classifier1 = LogisticRegression(random_state=0)
classifier1.fit(X_train, y_train)
startLR = time.time()
y_predL = classifier.predict(X_test)
endLR = time.time()

startLRboost = time.time()
classifierada = AdaBoostClassifier(random_state=0, base_estimator=classifier1, learning_rate=1)
classifierada.fit(X_train, y_train)
y_predboost = classifierada.predict(X_test)
endLRboost = time.time()

startLRbagger = time.time()
bagger = BaggingClassifier(base_estimator=classifier1)
bagger.fit(X_train, y_train)
y_predbagger = bagger.predict(X_test)
endLRbagger = time.time()

LRtime = round(endLR - startLR, 2)
LRboost = round(endLRboost - startLRboost, 2)
LRbag = round(endLRbagger - startLRbagger, 2)

print(LRtime, LRboost)

cmLR = confusion_matrix(y_test, y_predL)
cmLRboost = confusion_matrix(y_test, y_predL)
cmLRbag = confusion_matrix(y_test, y_predbagger)

y_predbagger = bagger.predict(X_test)
print("LR", "\n")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
print("LR boosted", "\n")
print(classification_report(y_test, y_predboost, target_names=['negative', 'positive']))
print("LR bagged", "\n")
print(classification_report(y_test, y_predbagger, target_names=['negative', 'positive']))

from sklearn.tree import DecisionTreeClassifier

StartDT = time.time()
classifier2 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier2.fit(X_train, y_train)
y_predD = classifier2.predict(X_test)
EndDT = time.time()

StartDTboost = time.time()
classifierada = AdaBoostClassifier(random_state=0, base_estimator=classifier2, learning_rate=1)
classifierada.fit(X_train, y_train)
y_predDTboosted = classifierada.predict(X_test)

EndDTboost = time.time()

StartDTbag = time.time()
bagger = BaggingClassifier(base_estimator=classifier2)
y_predDTbagged = classifierada.predict(X_test)
bagger.fit(X_train, y_train)
EndDTbag = time.time()

DTtime = round(EndDT - StartDT, 2)
DTboost = round(EndDTboost - StartDTboost, 2)
DTbag = round(EndDTbag - StartDTbag, 2)


cmDT = confusion_matrix(y_test, y_predD)
cmDTboost = confusion_matrix(y_test, y_predDTboosted)
cmDTbag = confusion_matrix(y_test, y_predDTbagged)

print("DT", "\n")
print(classification_report(y_test, y_predD, target_names=['negative', 'positive']))
print("DT boosted", "\n")
print(classification_report(y_test, y_predDTboosted, target_names=['negative', 'positive']))
print("DT bagged", "\n")
print(classification_report(y_test, y_predDTbagged, target_names=['negative', 'positive']))

from sklearn.ensemble import RandomForestClassifier

startRF = time.time()
classifier3 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier3.fit(X_train, y_train)
y_predR = classifier3.predict(X_test)
endRF = time.time()

startRFboost = time.time()
abc = AdaBoostClassifier(n_estimators=10, base_estimator=classifier3, learning_rate=1)
abc.fit(X_train, y_train)
y_predRboosted = abc.predict(X_test)
endRFboost = time.time()

startRFbag = time.time()
bagger = BaggingClassifier(base_estimator=classifier3)
bagger.fit(X_train, y_train)
y_predbagger = bagger.predict(X_test)
endRFbag = time.time()

RFtime = round(endRF - startRF, 2)
RFboost = round(endRFboost - startRFboost, 2)
RFbag = round(endRFbag - startRFbag, 2)

# Making the Confusion Matrix

cmRF = confusion_matrix(y_test, y_predR)
cmRFboost = confusion_matrix(y_test, y_predRboosted)
cmRFbag = confusion_matrix(y_test, y_predbagger)

print("RF", "\n")
print(classification_report(y_test, y_predR, target_names=['negative', 'positive']))
print("RF boosted", "\n")
print(classification_report(y_test, y_predRboosted, target_names=['negative', 'positive']))
print("RF bagged", "\n")
print(classification_report(y_test, y_predbagger, target_names=['negative', 'positive']))

RFtime = round(endRF - startRF, 2)
RFboost = round(endRFboost - startRFboost, 2)
RFbag = round(endRFbag - startRFbag, 2)

from sklearn.svm import SVC

Startsvc = time.time()
classifier = SVC(probability=True, kernel='linear')
classifier.fit(X_train, y_train)
y_predS = classifier.predict(X_test)

endsvc = time.time()

Startsvcboost = time.time()
abc = AdaBoostClassifier(base_estimator=classifier, learning_rate=1)
model = abc.fit(X_train, y_train)
y_predSboosted = abc.predict(X_test)
endsvcboost = time.time()

Startsvcbag = time.time()
bagger = BaggingClassifier(base_estimator=classifier)
bagger.fit(X_train, y_train)
y_predSbagged = bagger.predict(X_test)
endsvcbag = time.time()

svctime = round(endsvc - Startsvc, 2)
svcboosttime = round(endsvcboost - Startsvcboost, 2)
svcbagtime = round(endsvcbag - Startsvcbag, 2)

print("SVC", "\n")
print(classification_report(y_test, y_predS, target_names=['negative', 'positive']))
print("SVC boosted", "\n")
print(classification_report(y_test, y_predSboosted, target_names=['negative', 'positive']))
print("SVC bagged", "\n")
print(classification_report(y_test, y_predSbagged, target_names=['negative', 'positive']))

cmsvc = confusion_matrix(y_test, y_predS)
cmsvcboost = confusion_matrix(y_test, y_predSboosted)
cmsvcbag = confusion_matrix(y_test, y_predbagger)

from sklearn.neighbors import KNeighborsClassifier

startknn = time.time()
classifier4 = KNeighborsClassifier(n_neighbors=5, metric='cosine', p=2)
classifier4.fit(X_train, y_train)
y_predK = classifier4.predict(X_test)
endknn = time.time()

Knntime = round(endknn - startknn, 2)

# Predicting the Test set results

cmKnn = confusion_matrix(y_test, y_predK)

print("Kneigh", "\n")
print(classification_report(y_test, y_predK, target_names=['negative', 'positive']))

import matplotlib.pyplot as plt



svctime = round(endsvc - Startsvc)
svcboosttime = round(endsvcboost - Startsvcboost, 2)
svcbagtime = round(endsvcbag - Startsvcbag, 2)

index = 'Naive Bayes', 'Logistic Regression', 'SVC', 'Decision Tree', 'K-NN'
df = pd.DataFrame({"Base Estimator": [nbms, LRtime, svctime, DTtime, Knntime],
                   "Boosted Classifier": [boostedms, LRboost, svcboosttime, DTboost, 0],
                   "Bagged Classifier": [boostedms, LRboost, svcbagtime, DTboost, 0]}, index=index)
# df = df.reindex(reversed(sorted(df.columns)), axis = 1)

print(df)
ax = df.plot.barh()
ax.set_ylabel("Classifier", fontsize=20)
ax.set_xlabel("time(s)", fontsize=20)
ax.set_title("Run Time of Classifiers on Dataset1 (Binary-BOW)", fontsize=20)
plt.grid()
plt.show()
print(index)
ax = df.plot.barh()
print(df)

fig, (ax1, ax2, ax3, ax4, ax5, ax6,ax7,ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15) = plt.subplots(ncols=3, nrows=4, sharey=True)
sns.heatmap(cmNB, annot=True, ax=ax1, fmt='g', )
sns.heatmap(cmnNBboost, annot=True, ax=ax2, fmt='g')
sns.heatmap(cmNBbag, annot=True, ax=ax3, fmt='g')

sns.heatmap(cmLR, annot=True, ax=ax4, fmt='g')
sns.heatmap(cmLRboost, annot=True, ax=ax5, fmt='g')
sns.heatmap(cmLRbag, annot=True, ax=ax6, fmt='g')

sns.heatmap(cmDT, annot=True, ax=ax7, fmt='g')
sns.heatmap(cmDTboost, annot=True, ax=ax8, fmt='g')
sns.heatmap(cmDTbag, annot=True, ax=ax9, fmt='g')

sns.heatmap(cmRF, annot=True, ax=ax10, fmt='g', )
sns.heatmap(cmRFbag, annot=True, ax=ax11, fmt='g')
sns.heatmap(cmRFboost, annot=True, ax=ax12, fmt='g')

sns.heatmap(cmsvc, annot=True, ax=ax13, fmt='g')
sns.heatmap(cmsvcboost, annot=True, ax=ax14, fmt='g')
sns.heatmap(cmsvcbag, annot=True, ax=ax15, fmt='g')

ax1.set_xlabel('Predicted labels');
ax2.set_xlabel('Predicted labels');
ax1.set_ylabel('True labels');
ax1.set_title("Gaussian Naive Bayes");
ax2.set_title("Naive Bayes Boosted");
ax3.set_title("Naive Bayes Bagged")
ax1.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax2.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax1.yaxis.set_ticklabels(['Fake news', 'Real News']);

ax4.set_xlabel('Predicted labels');
ax5.set_xlabel('Predicted labels');
ax4.set_ylabel('True labels');
ax4.set_title("Logistic Regression");
ax5.set_title("Logistic Regression Boosted");
ax6.set_title("Logicstic Regression Bagged")
ax4.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax5.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax4.yaxis.set_ticklabels(['Fake news', 'Real News']);



ax7.set_xlabel('Predicted labels');
ax8.set_xlabel('Predicted labels');
ax7.set_ylabel('True labels');
ax7.set_title("Decision Tree");
ax8.set_title("Decision Tree Boosted");
ax9.set_title("Decision Tree Bagged")
ax7.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax8.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax7.yaxis.set_ticklabels(['Fake news', 'Real News']);


ax10.set_xlabel('Predicted labels');
ax11.set_xlabel('Predicted labels');
ax10.set_ylabel('True labels');
ax10.set_title("Random Forest");
ax11.set_title("Random forest Boosted");
ax12.set_title("Random forest Bagged")
ax10.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax8.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax7.yaxis.set_ticklabels(['Fake news', 'Real News']);



ax13.set_xlabel('Predicted labels');
ax14.set_xlabel('Predicted labels');
ax13.set_ylabel('True labels');
ax13.set_title("SVC");
ax15.set_title("SVC Boosted");
ax14.set_title("SVC  Bagged")
ax13.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax4.xaxis.set_ticklabels(['Fake News', 'Real News']);
ax13.yaxis.set_ticklabels(['Fake news', 'Real News']);


plt.show()

# df = df.sort_values('Amount', ascending=False)
