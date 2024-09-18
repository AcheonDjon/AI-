import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the data
spam = pd.read_csv('/workspaces/AI-/spam (1).csv', encoding='latin1')

# Assuming the CSV has columns 'label' and 'text'
X = spam.iloc[:, 1]  # Select the 'text' column (features)
y = spam.iloc[:, 0]  # Select the 'label' column (labels)

# Encode the labels as numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# print(X_train_transformed)

# Initialize the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train_transformed, y_train)

# Predict on the test set
y_pred = model.predict(X_test_transformed)

# Evaluate the model
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {score}')