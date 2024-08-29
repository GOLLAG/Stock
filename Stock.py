from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define the paragraph
paragraph = "I had an amazing experience at the new restaurant downtown. The food was exceptional, and the service was top-notch. I would definitely recommend it to anyone looking for a great meal!"

# Analyze the sentiment
sentiment_scores = sia.polarity_scores(paragraph)

# Print results
print("Sentiment Scores:", sentiment_scores)

# Interpret the results
compound_score = sentiment_scores['compound']
if compound_score >= 0.05:
    sentiment = "positive"
elif compound_score <= -0.05:
    sentiment = "negative"
else:
    sentiment = "neutral"

print(f"The sentiment is {sentiment}.")
