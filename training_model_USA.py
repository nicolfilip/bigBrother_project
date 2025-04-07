import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import random
import xgboost as xgb
from sklearn.metrics import ndcg_score

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
# data["eliminated"] = data["final_placement"].apply(lambda x: 1 if x > 1 else 0)
#
# X = data[["age", "avg_sentiment"]].dropna()
# y = data.loc[X.index, "eliminated"]
# מחשבים ציון דירוג – ככל שמקום הגמר גבוה יותר (כלומר 1, 2...), הציון יהיה גבוה יותר
data["rank_score"] = data["final_placement"].max() - data["final_placement"] + 1

# בחירת הפיצ'רים
X = data[["age", "avg_sentiment"]].dropna()
y = data.loc[X.index, "rank_score"]

# נחלק את הדאטה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# הכנה ל־DMatrix של XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# הגדרת הפרמטרים
params = {
    "objective": "rank:pairwise",
    "learning_rate": 0.1,
    "max_depth": 4,
    "n_estimators": 100
}

# אימון
model = xgb.train(params, dtrain, num_boost_round=100)

# חיזוי וציוני דירוג
y_pred = model.predict(dtest)

# מדידת איכות: normalized Discounted Cumulative Gain (nDCG)
print("nDCG:", ndcg_score([y_test], [y_pred]))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# print("\n Model accuracy: ", accuracy_score(y_test, y_pred))
# print("\nClassification report:")
# print(classification_report(y_test, y_pred))
# תחזית על כל המתמודדים
all_data_dmatrix = xgb.DMatrix(X)
all_data_preds = model.predict(all_data_dmatrix)

# נוסיף את התחזית לטבלה
data.loc[X.index, "predicted_score"] = all_data_preds

# דירוג מתמודדים מהכי צפוי לזכות ועד הכי פחות
ranking = data.loc[X.index].sort_values(by="predicted_score", ascending=False)
print(ranking[["full_name", "final_placement", "avg_sentiment", "predicted_score"]])
