import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Descriptive Statistics Functions
def calculate_headline_length(df):
    df['headline_length'] = df['headline'].apply(len)
    return df['headline_length'].describe()

def count_articles_per_publisher(df):
    return df['publisher'].value_counts()

def analyze_publication_dates(df):
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    df['publication_day'] = df['publication_date'].dt.day_name()
    return df['publication_day'].value_counts()

# Text Analysis Functions
def perform_sentiment_analysis(df):
    df['sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_category'] = pd.cut(df['sentiment'], bins=[-1, -0.05, 0.05, 1], labels=['Negative', 'Neutral', 'Positive'])
    return df['sentiment_category'].value_counts()

def perform_topic_modeling(df, n_topics=5):
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    vectorizer = CountVectorizer(stop_words=stop_words)
    data_vectorized = vectorizer.fit_transform(df['headline'])
    
    lda = LDA(n_components=n_topics, random_state=42)
    lda.fit(data_vectorized)
    
    return lda, vectorizer

def print_topics(lda_model, vectorizer, top_n=10):
    for idx, topic in enumerate(lda_model.components_):
        print(f"Topic {idx}:")
        print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-top_n - 1:-1]])

# Time Series Analysis Functions
def publication_frequency_over_time(df):
    df.set_index('publication_date', inplace=True)
    return df['headline'].resample('D').count()

def analyze_publishing_times(df):
    df['publication_hour'] = df.index.hour
    return df['publication_hour'].value_counts().sort_index()

# Publisher Analysis Functions
def analyze_publisher_domains(df):
    df['publisher_domain'] = df['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else x)
    return df['publisher_domain'].value_counts()

# Visualization Functions
def plot_bar(data, title, xlabel, ylabel, rotation=90):
    plt.figure(figsize=(10,6))
    sns.barplot(x=data.index, y=data.values, palette='viridis')
    plt.xticks(rotation=rotation)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_histogram(data, title, xlabel, ylabel):
    plt.figure(figsize=(12,8))
    data.hist(bins=50, grid=False, color='#86bf91', zorder=2, rwidth=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_time_series(data, title, xlabel, ylabel):
    plt.figure(figsize=(12,6))
    data.plot()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_line(data, title, xlabel, ylabel):
    plt.figure(figsize=(10,6))
    sns.lineplot(x=data.index, y=data.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
