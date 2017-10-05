import pandas as pd
import sklearn.pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_csv("crimes_processed.csv")
for t in df.TYPE.unique():
    print(t,len(df[df.TYPE == t]))

X_columns = [i for i in df.columns if i != "TYPE"]
y_columns = ["TYPE"]

X = df.loc[:,X_columns]
y = df.TYPE
labelEnc = LabelEncoder()
X.loc[:,"NEIGHBOURHOOD"] = labelEnc.fit_transform(X.NEIGHBOURHOOD)
X.loc[:,"WEEKDAY"] = labelEnc.fit_transform(X.WEEKDAY)
X.loc[:,"BLOCK"] = labelEnc.fit_transform(X.BLOCK)
y = labelEnc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(max_depth=50, min_samples_split=20, n_estimators=100, n_jobs=8)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
report = classification_report(y_train, y_pred)
print(report)
print("Train score: {}".format(clf.score(X_train, y_train)))
print("Test score: {}".format(clf.score(X_test, y_test)))

