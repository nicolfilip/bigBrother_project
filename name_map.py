import pandas as pd

# קריאת קובץ הגרף באנגלית
df_graph = pd.read_csv("graph_output/graph_eng")  # הוספתי סיומת .csv

# מיפוי שמות מאנגלית לעברית (מכסה את כל מי שהופיע בקובץ)
name_map = {
    "Maor Bruchman": "מאור ברוכמן",
    "May Erev": "מאי ערב",
    "Dror Clear": "דרור קליר",
    "Shani Edri": "שני אדרי",
    "Tirza Cohen": "תרצה כהן",
    "Erez Isakov": "ארז איסקוב",
    "Lauren Gozlan": "לורן גוזלן",
    "Lauren Goszlan": "לורן גוזלן",
    "Erez Mughrabi": "ארז מוגרבי",
    "Shelly Sandrov": "שלי סנדרוב",
    "Shelly Ebenstein": "שלי אבנשטיין",
    "Anna Sophia Kalman": "אנה סופיה קלמן",
    "Chen Kraunick": "חן קרוניק",
    "Yuval Levi": "יובל לוי",
    "Elaya Hof": "אלאיה הוף",
    "Netanel Dahan": "נתנאל דהאן",
    "Yarden Edri": "ירדן אדרי"
}

# החלת המיפוי
df_graph["from"] = df_graph["from"].map(name_map).fillna(df_graph["from"])
df_graph["to"] = df_graph["to"].map(name_map).fillna(df_graph["to"])

# שמירה
df_graph.to_csv("graph_output/graph_heb.csv", index=False, encoding="utf-8-sig")

print("✅ done – graph_output/graph_heb.csv")
