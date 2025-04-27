import pandas as pd
import tweepy
import os
import time
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# הגדרת משתנים
load_dotenv()
bearer_token = os.getenv("BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

query_for_ISRAEL = '"האח הגדול" -is:retweet lang:he'
tweet_count_ISRAEL = 5 # המינימום המותר הוא 10, המקסימום 100

# שליפת ציוצים עם טיפול ב-Rate Limit
def safe_search_tweets(query, count):
    while True:
        try:
            return client.search_recent_tweets(
                query=query,
                max_results=count,
                tweet_fields=["created_at", "text", "public_metrics"]
            )
        except tweepy.TooManyRequests:
            print("⚠️ Rate limit hit. Waiting 15 minutes...")
            time.sleep(900)
        except Exception as e:
            print("שגיאה:", e)
            break

# הרצת השאילתה
tweets = safe_search_tweets(query_for_ISRAEL, tweet_count_ISRAEL)

# עיבוד הציוצים
tweet_data = []
if tweets and tweets.data:
    for tweet in tweets.data:
        tweet_data.append([
            tweet.created_at,
            tweet.text,
            tweet.public_metrics["like_count"],
            tweet.public_metrics["retweet_count"]
        ])

    df = pd.DataFrame(tweet_data, columns=["timestamp", "text", "likes", "retweets"])

    # ניתוח סנטימנט
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["text"].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

    # שמירת קובץ
    df.to_csv("big_brother_tweets_ISRAEL.csv", index=False, encoding="utf-8")
    print("🎉 success! נשמר קובץ: big_brother_tweets_ISRAEL.csv")

else:
    print("😕 לא נמצאו ציוצים מתאימים.")
