import csv
import pandas as pd
import tweepy
import os
from dotenv import load_dotenv


file_path = "big_brother_data.csv"
try:
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

print("Sample data:")
print(data.head())


def load_env(file_path=".env"):
    try:
        with open(file_path) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                key, value = line.strip().split("=", 1)
                os.environ[key] = value
    except FileNotFoundError:
        print(f"Error: .env file not found at {file_path}")
        exit()


load_dotenv()
bearer_token = os.getenv("BEARER_TOKEN")

# register to twitter API V2
client = tweepy.Client(bearer_token=bearer_token)

# search posts on big brother
query = "Big Brother -is:retweet lang:en"
tweet_count = 10

# שליפת הפוסטים
tweets = client.search_recent_tweets(query=query, max_results=tweet_count, tweet_fields=["created_at", "text", "public_metrics"])

# עיבוד הנתונים ושמירתם
tweet_data = []
for tweet in tweets.data:
    tweet_data.append([
        tweet.created_at, tweet.text,
        tweet.public_metrics["like_count"],
        tweet.public_metrics["retweet_count"]
    ])

# יצירת DataFrame ושמירתו
df = pd.DataFrame(tweet_data, columns=["timestamp", "text", "likes", "retweets"])
df.to_csv("big_brother_tweets_v2.csv", index=False, encoding="utf-8")

print(f"✅ {len(df)} tweets saved to 'big_brother_tweets_v2.csv'")
print(df.head())