import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import random

name = yuval
contestants = pd.read_csv("big_brother_usa.csv")
tweets = pd.read_csv("big_brother_tweets_USA.csv")

#the full name consists in the csv first and last name
contestants["full_name"] = contestants["first"].astype(str) + " " + contestants["last"].astype(str)

#setiment for every contestant. between -1 to 1. close to 1- positive, close to -1- negative
tweets["sentiment"] = tweets["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)


#match the twitts to the contestants by name
def match_contestant(text, contestant_names):
    for name in contestant_names:
        if pd.isna(name):
            continue
        #turn the name to lower and the twit
        if str(name).lower() in str(text).lower():
            return name
    return None


tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, contestants["full_name"]))
tweets_filtered = tweets.dropna(subset=["username"])

avg_sentiment = tweets_filtered.groupby("username")["sentiment"].mean().reset_index()
avg_sentiment.columns = ["full_name", "avg_sentiment"]

#merge the name and the avg sentiment
data = contestants.merge(avg_sentiment, on="full_name", how="left")
data["avg_sentiment"] = data["avg_sentiment"].fillna(0)

#add col. who won is 1. the contestants that finished not in the first place tag 0
data["eliminated"] = data["final_placement"].apply(lambda x: 1 if x > 1 else 0)

X = data[["age", "avg_sentiment"]].dropna()
y = data.loc[X.index, "eliminated"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n Model accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
