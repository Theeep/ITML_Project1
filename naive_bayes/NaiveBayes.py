import random
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.pipeline
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report

gnb = GaussianNB()
# Reading csv
data = pd.read_csv("crimes_processed.csv", nrows=50000)
target = data["TYPE"]
# Numberizing data
neighbour = LabelEncoder()
data["NEIGHBOURHOOD"] = neighbour.fit(
    data["NEIGHBOURHOOD"]).transform(data["NEIGHBOURHOOD"])
weekday = LabelEncoder()
data["WEEKDAY"] = weekday.fit(data["WEEKDAY"]).transform(data["WEEKDAY"])
hundred = LabelEncoder()
data["HUNDRED"] = hundred.fit(data["HUNDRED"]).transform(data["HUNDRED"])

cols = [x for x in data.columns if x not in ["TYPE", "BLOCK"]]

targetFitted = LabelEncoder().fit_transform(target)
steps = [
    ("naiveBayes_classifier", gnb)
]

parameters = dict(
    naiveBayes_classifier__priors=[
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
            1 / 8],        # All types weighted equally
        # Majority class weighted heavier
        [1 / 16, 1 / 8, 1 / 8, 1 / 8, 2 / 8, 1 / 16, 1 / 8, 1 / 8],
        None]
)

random.seed()
seed = random.randint(1, 100)


X_train, X_test, y_train, y_test = train_test_split(
    data[cols], targetFitted, test_size=0.2, random_state=seed)

nb_pipeline = sklearn.pipeline.Pipeline(steps)
grid_search = GridSearchCV(nb_pipeline, param_grid=parameters, n_jobs=4)
grid_search.fit(X_train, y_train)
train_prediction = grid_search.predict(X_train)
test_prediction = grid_search.predict(X_test)


train_report = classification_report(y_train, train_prediction)
test_report = classification_report(y_test, test_prediction)
print(train_report)
print(test_report)
print(grid_search.best_estimator_.get_params()["steps"])

result_df = pd.DataFrame(grid_search.cv_results_)
#print(result_df)
result_df.to_csv("naive_results.csv", encoding="utf-8", index=False)
