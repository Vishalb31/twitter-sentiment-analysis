🐦 Twitter Sentiment Analysis | NLP Project
This project focuses on performing Sentiment Analysis on Twitter data from airline customers using Natural Language Processing (NLP) techniques and Machine Learning models. The project includes comprehensive data cleaning, visualization, classification, clustering, and insight generation using Python.
________________________________________
📌 Project Overview
•	Objective: To classify tweets into Positive, Negative, or Neutral sentiments.
•	Dataset: Airline Twitter Sentiment Dataset (Kaggle)
•	Approach:
o	Clean and preprocess raw tweet text.
o	Apply tokenization and stop-word removal.
o	Train multiple classifiers (Naive Bayes, Logistic Regression, Random Forest).
o	Use TF-IDF for text vectorization.
o	Evaluate models with confusion matrices and accuracy scores.
o	Visualize sentiment distribution and generate Word Clouds.
o	Cluster tweets using K-Means algorithm.
o	Predict sentiment for custom tweets.
________________________________________
🛠️ Tools & Technologies Used
Area	Tools/Libraries
Programming Language	Python
Data Manipulation	Pandas, NumPy
Visualization	Matplotlib, Seaborn, Plotly
Text Processing	NLTK, re, string
ML Models	Scikit-learn (Naive Bayes, Logistic Regression, Random Forest)
Vectorization	TF-IDF Vectorizer
NLP Tasks	Tokenization, Stopword Removal, Word Cloud
Clustering	KMeans
IDE	Google Colab
________________________________________
🧹 Data Cleaning Steps
•	Removed URLs, mentions (@user), hashtags (#), punctuation, and extra whitespace.
•	Converted all text to lowercase.
•	Removed common English stop-words using nltk.corpus.stopwords.
•	Tokenized the text using nltk.word_tokenize.

python
CopyEdit
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)
________________________________________
📊 Visualizations & Dashboards
•	Sentiment Distribution: Bar Chart & Pie Chart
•	Word Clouds:
o	Overall
o	Positive Tweets
o	Negative Tweets
•	Scatter Plot of Sentiment vs Tweet Length
•	Correlation Heatmaps
•	K-Means Clustering visualization
________________________________________
🤖 Machine Learning Models Used
Model	Description
Naive Bayes	Baseline text classifier using probabilities
Logistic Regression	For binary/multi-class sentiment classification
Random Forest	Ensemble model for better performance
📈 Model Evaluation
•	Confusion Matrix
•	Accuracy Score
•	Classification Report (Precision, Recall, F1-Score)
________________________________________
🔍 Extra Features
•	Predict sentiment on custom/new tweets.
•	Display top 5 positive and top 5 negative tweets.
•	Clustering of tweets using K Means to find tweet groupings.
•	Tokenization for better model training.
________________________________________
✅ How to Run
1.	Clone the repository
2.	Open TSA.ipynb in Google Colab
3.	Upload the dataset Tweets.csv
4.	Run all cells and modify the custom tweet text for predictions if desired
