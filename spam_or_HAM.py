from sklearn.calibration import LabelEncoder
import tensorflow as tf
import pandas as pd
import xgboost as xgb
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
spam = pd.read_csv('/Users/zebra/Documents/projects/AI/spam (1).csv', encoding= 'latin1')

y = spam.iloc[:, 0]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X = spam.iloc[:, 1]


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)

X_test_transformed = vectorizer.transform(X_test)


print(y_encoded)

# model = xgb.XGBClassifier()
# model.fit(X_train_transformed,y_encoded)

# y_pred = model.predict(X_test)
# score = accuracy_score(y_test, y_pred)

# print(score)