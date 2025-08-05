import pandas as pd

df_graph = pd.read_csv("graph_output/season15/graph_eng_15")
df_graph_12 = pd.read_csv("graph_output/season12/graph_eng_12.csv")

name_map_12 = {
    "kazem halila": "קאזם חלילה",
    "dina samhi": "דינה ערמי שמחי",
    "dian shwartz": "דיאן שוורץ",
    "netanel rodnitzki": "נתנאל רודניצקי",
    "marina kuznetsova": "מרינה קוזנצובה",
    "eliav tati": "אליאב טעטי",
    "riwaa raslan": "ריוואה רסלאן",
    "shahaf raz": "שחף רז",
    "sharin avraham": "שרין טובה אברהם",
    "ofek levi": "אופק לוי",
    "bar cohen": "בר נעה כהן",
    "daniel malka": "דניאל מלכה",
    "rachel borta": "רחל בורטה",
    "ilana taranenko": "אילנה טרננקו",
    "diana taranenko": "דיאנה טרננקו",
    "sharon benifusi": "שרון בניפוסי",
    "talia ovadia": "טליה עובדיה",
    "omri alfia": "עמרי אלפיה",
    "dana amsalem": "דנה אמסלם",
    "david matzri": "דוד מהצרי",
    "lia tai aharoni": "ליה טאי אהרוני",
    "liran rosen": "לירן רוזן",
    "moshe kugman": "משה קוגמן",
    "dror rokenstein": "דרור רוקנשטיין"
}


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
df_graph_12["from"] = df_graph_12["from"].map(name_map_12).fillna(df_graph_12["from"])
df_graph_12["to"] = df_graph_12["to"].map(name_map_12).fillna(df_graph_12["to"])

df_graph_12.to_csv("graph_output/graph_heb_12.csv", index=False, encoding="utf-8-sig")

print("✅ done – graph_output/graph_heb_12.csv")

df_graph["from"] = df_graph["from"].map(name_map).fillna(df_graph["from"])
df_graph["to"] = df_graph["to"].map(name_map).fillna(df_graph["to"])

# save
df_graph.to_csv("graph_output/graph_heb.csv", index=False, encoding="utf-8-sig")

print("✅ done – graph_output/graph_heb_15.csv")
