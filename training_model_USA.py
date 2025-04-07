import sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

participants = pd.read_csv("big_brother_usa.csv")
tweets = pd.read_csv("big_brother_tweets_USA.csv")
participants["full name"] = participants["first"] + " " + participants["last"]
tweets["sentiment"] = tweets["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)


def matching_name_in_tweets(text, names):
    for name in names:
        if pd.isna(name):
            continue
        if str(name).lower() in str(text).lower():
            return name
        return None


tweets["userName"] = tweets["text"].apply(lambda x: matching_name_in_tweets(str(x), participants["full name"]))
filtered_tweets = tweets.dropna(
    subset=["userName"])  #check every row in tweets, if the value user name is Nan- then drop

avg_sentiment = filtered_tweets.groupby("userName")["sentiment"].mean()
avg_sentiment.columns = ["avg_sentiment"].fillna(0)

mergeSentimentToParticipants = participants.merge(avg_sentiment, on="full name", how="left")
mergeSentimentToParticipants["eliminated"] = mergeSentimentToParticipants["final_placement"].apply(
    lambda x: 1 if x > 1 else 0)
orderOfElimination = mergeSentimentToParticipants.sort_values("final_placement").reset_index(drop=True)
num_weeks = 7
divide_participants = len(orderOfElimination) // num_weeks
allTrue = []
allPred = []
for i in range(num_weeks):
    start = i * divide_participants
    end = start + divide_participants
    week_data = orderOfElimination.iloc[start:end]

    X = week_data[["age", "avg_sentiment"]].dropna()
    y = week_data.loc[X.index, "eliminated"]

    if len(y.unique()) < 2:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    allTrue.extend(y_test)
    allPred.extend(y_pred)

# Accuracy and classification report
print("\nModel accuracy:", accuracy_score(allTrue, allPred))
print("\nClassification report:")
print(classification_report(allTrue, allPred))

# Confusion Matrix
conf_matrix = confusion_matrix(allTrue, allPred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Precision – כמה תחזיות של "הודח" היו נכונות
#
# Recall – כמה מתוך כל המודחים זוהו
#
# F1-score – ממוצע בין שניהם