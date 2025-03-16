import pandas as pd
import tweepy
import os
from dotenv import load_dotenv
import docx

# access to the word file that we did on the Participants of big brother Israel
file_path = "big_brother_israel_new.docx"
doc = docx.Document(file_path)

# check the tables in the file
tables = doc.tables
if not tables:
    print("there is no tables in the file")
    exit()

# read the tables
all_data = []
for table_index, table in enumerate(tables):
    table_data = []

# read the lines
    for row in table.rows:
        table_data.append([cell.text.strip() for cell in row.cells])  # delete un relevant spaces
    df = pd.DataFrame(table_data)
# add a col that will mention from which number of table the data came from
    df["table_index"] = table_index + 1
    all_data.append(df)
final_df = pd.concat(all_data, ignore_index=True)
# save to CSV
csv_file_path = "big_brother_israel.csv"
final_df.to_csv(csv_file_path, index=False, encoding="utf-8")


# -------------------------------------------------------------------------------

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


# search posts on big brother
query = "Big Brother -is:retweet lang:en"
tweet_count = 100

# take out the posts by the query big brother, by the number of tweets that we want and by the fields
tweets = client.search_recent_tweets(query=query, max_results=tweet_count, tweet_fields=["created_at", "text", "public_metrics"])

# processing and save data
tweet_data = []
for tweet in tweets.data:
    tweet_data.append([
        tweet.created_at, tweet.text,
        tweet.public_metrics["like_count"], # number of likes
        tweet.public_metrics["retweet_count"]
    ])

# make table from the data
df = pd.DataFrame(tweet_data, columns=["timestamp", "text", "likes", "retweets"])
df.to_csv("big_brother_tweets_v2.csv", index=False, encoding="utf-8")

