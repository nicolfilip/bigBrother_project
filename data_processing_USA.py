import pandas as pd
import tweepy
import os
from dotenv import load_dotenv
import time
from tweepy.errors import TooManyRequests


def safe_search_tweets(query, count):
    while True:
        try:
            tweets = client.search_recent_tweets(
                query=query,
                max_results=count,
                tweet_fields=["created_at", "text", "public_metrics"]
            )
            return tweets
        except TooManyRequests:
            print("⚠️ Twitter rate limit exceeded. Waiting 15 minutes before retrying...")
            time.sleep(900)


file_path = "big_brother_usa.csv"
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
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# search posts on big brother USA
query_for_USA = "Big Brother -is:retweet lang:en"
tweet_count_USA = 100

# take out the posts by the query big brother, by the number of tweets that we want and by the fields
tweets = safe_search_tweets(query_for_USA, tweet_count_USA)

# processing and save data
tweet_data_USA = []
for tweet in tweets.data:
    tweet_data_USA.append([
        tweet.created_at, tweet.text,
        tweet.public_metrics["like_count"],  # number of likes
        tweet.public_metrics["retweet_count"]
    ])

# make table from the data
df_usa = pd.DataFrame(tweet_data_USA, columns=["timestamp", "text", "likes", "retweets"])
df_usa.to_csv("big_brother_tweets_USA.csv", index=False, encoding="utf-8")
