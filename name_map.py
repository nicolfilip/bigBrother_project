import pandas as pd

# קריאת קובץ הגרף באנגלית
df_graph = pd.read_csv("graph_output/Untitled spreadsheet - Sheet1.csv")

# מיפוי שמות מאנגלית לעברית
name_map = {
    "Maor Bruchman": "מאור ברוכמן",
    "May erev": "מאי ערב",
    "Dror Clear": "דרור קליר",
    "Shani Edri": "שני אדרי",
    "Tirza Cohen": "תרצה כהן",
    "Erez Isakov": "ארז איזקוב",
    "Lauren Gozlan": "לורן גוזלן",
    "Lauren Goszlan": "לורן גוזלן",
    "Erez Mughrabi": "ארז מוגרבי",
    "Shelly Sandrov": "שלי סנדרוב",
    "Shelly Ebenstein": "שלי אבן־שטיין",
    "Anna Sophia Kalman": "אנה סופיה קלמן",
    "Chen Kraunick": "חן קרוניק"
}

df_graph["from"] = df_graph["from"].map(name_map).fillna(df_graph["from"])
df_graph["to"] = df_graph["to"].map(name_map).fillna(df_graph["to"])

df_graph.to_csv("graph_output/graph_heb.csv", index=False, encoding="utf-8-sig")

print("✅ done- graph_output/graph_heb.csv")
