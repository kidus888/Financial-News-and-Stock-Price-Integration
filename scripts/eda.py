import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Function to load data
def load_data(filepath):
    return pd.read_csv(filepath)

# Descriptive Statistics
def headline_length_statistics(df):
    df['headline_length'] = df['headline'].apply(len)
    return df['headline_length'].describe()

def plot_headline_length_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['headline_length'], bins=30, kde=True)
    plt.title('Distribution of Headline Lengths')
    plt.show()

def count_articles_per_publisher(df):
    return df['publisher'].value_counts()

def plot_articles_per_publisher(df):
    publisher_counts = df['publisher'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=publisher_counts.index, y=publisher_counts.values)
    plt.title('Number of Articles per Publisher')
    plt.xticks(rotation=90)
    plt.show()

def publication_date_trends(df):
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    df['publication_day'] = df['publication_date'].dt.day_name()
    return df

def plot_publication_day_distribution(df):
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x='publication_day', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('Articles Published by Day of the Week')
    plt.show()

def plot_monthly_trend(df):
    df['publication_month'] = df['publication_date'].dt.to_period('M')
    monthly_trend = df['publication_month'].value_counts().sort_index()
    plt.figure(figsize=(14, 6))
    monthly_trend.plot(kind='line')
    plt.title('Publication Trend Over Time')
    plt.show()

# Text Analysis
def perform_sentiment_analysis(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df['sentiment']

def plot_sentiment_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], bins=30, kde=True)
    plt.title('Sentiment Distribution of Headlines')
    plt.show()

def generate_wordcloud(df):
    wordcloud = WordCloud(stopwords='english', background_color='white').generate(' '.join(df['headline']))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Time Series Analysis
def plot_publication_frequency(df):
    plt.figure(figsize=(14, 6))
    df.set_index('publication_date').resample('D').size().plot()
    plt.title('Daily Publication Frequency')
    plt.show()

def plot_publishing_time_distribution(df):
    df['publication_hour'] = df['publication_date'].dt.hour
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x='publication_hour')
    plt.title('Articles Published by Hour')
    plt.show()

# Publisher Analysis
def analyze_publisher_contribution(df):
    top_publishers = df['publisher'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_publishers.values, y=top_publishers.index)
    plt.title('Top 10 Publishers by Article Count')
    plt.show()

def analyze_publisher_domains(df):
    df['domain'] = df['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else x)
    domain_counts = df['domain'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=domain_counts.head(10).values, y=domain_counts.head(10).index)
    plt.title('Top 10 Domains by Article Count')
    plt.show()
