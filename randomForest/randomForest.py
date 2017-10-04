import pandas as pd
import sklearn.pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


df = pd.read_csv("crimesProcessed.csv", nrows=1000)


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

clf = RandomForestClassifier(n_jobs=8)

steps = [
    ('random_forest', clf),
]

parameters = dict(
    #random_forest__n_estimators=[10,20,50,100],
    random_forest__n_estimators=[10,100],
    #random_forest__max_depth=[20,50,100,200],
    random_forest__max_depth=[20,50],
    #random_forest__min_samples_split=[20,50,100,200,500,1000],
    random_forest__min_samples_split=[200,500],
    #random_forest__max_leaf_nodes=[100,300,600,1000,None],
    random_forest__max_leaf_nodes=[600,1000,None]
)

pipeline = sklearn.pipeline.Pipeline(steps)

cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=8)
#cv.fit(X_train,y_train)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_train)
report = classification_report(y_train, y_pred)
print(report)
print(pd.DataFrame(cv.cv_results_))

print("Best Parameters:",cv.best_params_)
print("Best score:",cv.best_score_)
