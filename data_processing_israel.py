import docx
<<<<<<< HEAD
from docx import Document
=======
>>>>>>> 3d3b309d6ef8b154136d206b0fbe45f05e866f51
import pandas as pd
import tweepy
import os
from dotenv import load_dotenv
import time
from tweepy.errors import TooManyRequests


# def safe_search_tweets(query, count):
#     while True:
#         try:
#             tweets = client.search_recent_tweets(
#                 query=query,
#                 max_results=count,
#                 tweet_fields=["created_at", "text", "public_metrics"]
#             )
#             return tweets
#         except TooManyRequests:
#             print("⚠️ Twitter rate limit exceeded. Waiting 15 minutes before retrying...")
#             time.sleep(900)


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
# save to CSV
csv_file_path = "big_brother_israel.csv"
final_df.to_csv(csv_file_path, index=False, encoding="utf-8")

df_cleaned = pd.read_csv(csv_file_path, header=1)
df_cleaned = df_cleaned[df_cleaned["שם מלא"].notna()]
df_cleaned["גיל"] = pd.to_numeric(df_cleaned["גיל"], errors="coerce")
df_cleaned["Days in game"] = pd.to_numeric(df_cleaned["Days in game"], errors="coerce")

df_cleaned.to_csv("big_brother_israel_cleaned.csv", index=False, encoding="utf-8")



# load_dotenv()
# bearer_token = os.getenv("BEARER_TOKEN")
#
# # register to twitter API V2
# client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
# # search posts on big brother ISRAEL
# query_for_ISRAEL = '"האח הגדול" -is:retweet lang:he'
# tweet_count_ISRAEL = 1
#
# # take out the posts by the query big brother, by the number of tweets that we want and by the fields
# tweets = client.search_recent_tweets(query=query_for_ISRAEL, max_results=tweet_count_ISRAEL, tweet_fields=["created_at", "text", "public_metrics"])
#
# # processing and save data
# tweet_data_ISRAEL = []
# for tweet in tweets.data:
#     (tweet_data_ISRAEL.append([
#         tweet.created_at, tweet.text,
#         tweet.public_metrics["like_count"],  # number of likes
#         tweet.public_metrics["retweet_count"]
#     ]))
#
# # make table from the data
# df_israel = pd.DataFrame(tweet_data_ISRAEL, columns=["timestamp", "text", "likes", "retweets"])
# df_israel.to_csv("big_brother_tweets_ISRAEL.csv", index=False, encoding="utf-8")

print("success")
