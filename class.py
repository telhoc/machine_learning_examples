import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


# 1. Read the Data
train = pd.read_csv('train.csv', delimiter='\t', header=None, names=['text', 'label'])
test = pd.read_csv('test.csv', delimiter='\t', header=None, names=['text', 'label'])

# 2. Preprocess the Data
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train['text'])
X_test = vectorizer.transform(test['text'])

y_train = train['label']
y_test = test['label']

# 3. Train a Model
#clf = LogisticRegression(max_iter=1000)
#clf = SVC(max_iter=1000)
#clf = RandomForestClassifier()
clf = GradientBoostingClassifier()
#clf = MultinomialNB(max_iter=1000)
#clf = KNeighborsClassifier(max_iter=1000)
#clf = SGDClassifier(max_iter=1000)
clf.fit(X_train, y_train)

# 4. Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

