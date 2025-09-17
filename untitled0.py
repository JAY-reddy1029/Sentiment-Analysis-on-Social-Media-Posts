import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\pvjay\Desktop\projects\done_by_me\Sentiment_Analysis_on_Social_Media_Posts\data-set.csv",encoding="latin-1", header=None)


# Assign column names (since dataset has none)

df.columns = ["target", "ids", "date", "flag", "user", "text"]
df = df[["target", "text"]]
df["target"] = df["target"].map({0: "Negative", 2: "Neutral", 4: "Positive"})

# Preview first rows
print(df.head())
print(df.shape)
print(df["target"].value_counts())

import re

def clean_tweet(text):
    text = str(text).lower()                          # lowercase
    text = re.sub(r'http\S+|www.\S+', '', text)       # remove urls
    text = re.sub(r'@\w+', '', text)                  # remove mentions
    text = re.sub(r'#\w+', '', text)                  # remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)              # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()          # remove extra spaces
    return text

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_tweet)

print(df[["text", "clean_text"]].head(10))

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (only once)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["clean_text"].apply(preprocess_text)

print(df[["text", "clean_text"]].head(10))


import matplotlib.pyplot as plt
import seaborn as sns

#Check how many tweets are Positive, Negative, Neutral.
plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df, palette="Set2")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.show()

#Let’s see the most common words in positive vs. negative tweets.
from wordcloud import WordCloud

# Positive tweets
positive_text = " ".join(df[df["target"]=="Positive"]["clean_text"])
wordcloud_pos = WordCloud(width=800, height=400, background_color="white").generate(positive_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Tweets Word Cloud")
plt.show()

# Negative tweets
negative_text = " ".join(df[df["target"]=="Negative"]["clean_text"])
wordcloud_neg = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(negative_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Tweets Word Cloud")
plt.show()

#Most Frequent Words
from collections import Counter

def get_top_words(sentiment, n=20):
    words = " ".join(df[df["target"]==sentiment]["clean_text"]).split()
    return Counter(words).most_common(n)

print("Top Positive Words:", get_top_words("Positive"))
print("Top Negative Words:", get_top_words("Negative"))

'''
#Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert tweets into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
X = vectorizer.fit_transform(df["clean_text"])

# Target labels
y = df["target"]

#Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train a Classical ML Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#Check how well your model performs using metrics like accuracy, F1-score, and confusion matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) '''

from sklearn.model_selection import GridSearchCV


#Feature Extraction (TF-IDF)

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert tweets into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # You can increase if needed
X = vectorizer.fit_transform(df["clean_text"])

# Target labels
y = df["target"]

# Train/Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM Model
from sklearn.svm import LinearSVC

# LinearSVC is faster than SVC for large datasets
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'loss': ['hinge', 'squared_hinge'],
    'max_iter': [5000,10000]
}

grid = GridSearchCV(estimator=LinearSVC(class_weight='balanced',random_state=42), 
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    cv=3, 
                    verbose=2, 
                    n_jobs=1)

# Fit GridSearchCV instead of plain SVM
grid.fit(X_train, y_train)

# Get the best model
svm_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validated Accuracy:", grid.best_score_)

# Make predictions
y_pred = svm_model.predict(X_test)

#Evaluate Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Accuracy & Classification Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative","Neutral","Positive"],
            yticklabels=["Negative","Neutral","Positive"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("SVM Confusion Matrix")
plt.show()

#Save Your Model
import pickle

# Save the trained SVM model
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

# Save the TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")

# --------------------------
# Load later when needed
with open("svm_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)

# Example test
sample = ["This movie was absolutely fantastic! I loved it."]
sample_tfidf = loaded_vectorizer.transform(sample)
print("Sample Prediction:", loaded_model.predict(sample_tfidf)[0])

import os
print(os.getcwd())

































































