from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is a sample document.",
    "This document is another example.",
    "We have yet another example here."
]

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Transform the documents into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense format and print
dense_matrix = tfidf_matrix.todense()

print("TF-IDF Matrix:\n", dense_matrix)
print("\nFeature Names:\n", feature_names)
