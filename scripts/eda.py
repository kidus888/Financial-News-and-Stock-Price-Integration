import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

def load_data(file_path):
    return pd.read_csv(file_path)

# Descriptive Statistics
def descriptive_stats(df):
    # Textual lengths
    df['headline_length'] = df['headline'].apply(len)
    print(df['headline_length'].describe())

    # Count the number of articles per publisher
    publisher_counts = df['publisher'].value_counts()
    print(publisher_counts)

    # Analyze publication dates
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    date_counts = df['publication_date'].value_counts().sort_index()
    plt.figure(figsize=(12,6))
    sns.lineplot(x=date_counts.index, y=date_counts.values)
    plt.title("Article Counts Over Time")
    plt.show()

# Text Analysis
def sentiment_analysis(df):
    df['sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)
    return df

def topic_modeling(df, n_topics=5):
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df['headline'])
    print("Topic modeling placeholder")
    return None

# Time Series Analysis
def time_series_analysis(df):
    df['publication_hour'] = df['publication_date'].dt.hour
    time_series_counts = df.groupby('publication_hour').size()
    plt.figure(figsize=(12,6))
    sns.lineplot(x=time_series_counts.index, y=time_series_counts.values)
    plt.title("Article Publication by Hour")
    plt.show()

# Publisher Analysis
def publisher_analysis(df):
    publisher_counts = df['publisher'].value_counts()
    print("Top publishers:\n", publisher_counts.head())
    df['publisher_domain'] = df['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else x)
    domain_counts = df['publisher_domain'].value_counts()
    print("Top domains:\n", domain_counts.head())
