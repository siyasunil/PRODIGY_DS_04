import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


# Load Data

df = pd.read_csv("twitter_training.csv", header=None, names=['id', 'topic', 'sentiment', 'text'])

# Basic info
print("Dataset shape:", df.shape)
print(df.head())


# 1. Overall Sentiment Distribution

sentiment_counts = df['sentiment'].value_counts()
print("\nSentiment Counts:\n", sentiment_counts)

# Bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'], startangle=90)
plt.title("Sentiment Percentage")
plt.show()


# 2. Sentiment Distribution by Topic (Brands)

topic_sentiment = df.groupby(['topic', 'sentiment']).size().unstack(fill_value=0)

# Heatmap to show sentiment patterns across topics
plt.figure(figsize=(12, 8))
sns.heatmap(topic_sentiment, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Sentiment Distribution by Topic")
plt.ylabel("Topic")
plt.xlabel("Sentiment")
plt.show()

# Stacked Bar Chart for better visualization
topic_sentiment.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='coolwarm')
plt.title("Stacked Sentiment Analysis by Topic")
plt.ylabel("Count")
plt.xlabel("Topic")
plt.legend(title='Sentiment')
plt.show()


# 3. WordCloud for Positive & Negative Sentiments

stopwords = set(STOPWORDS)

# Positive WordCloud
positive_text = " ".join(df[df['sentiment'] == 'Positive']['text'].astype(str))
wc_pos = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap='Greens').generate(positive_text)

plt.figure(figsize=(10, 6))
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Positive Sentiment")
plt.show()

# Negative WordCloud
negative_text = " ".join(df[df['sentiment'] == 'Negative']['text'].astype(str))
wc_neg = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap='Reds').generate(negative_text)

plt.figure(figsize=(10, 6))
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Negative Sentiment")
plt.show()


print("Key Insights:")
print("- Most common topics with positive/negative reactions are shown in the heatmap.")
print("- Overall sentiment distribution shows user attitudes toward brands.")
print("- WordCloud highlights frequently used terms in Positive vs Negative posts.")
