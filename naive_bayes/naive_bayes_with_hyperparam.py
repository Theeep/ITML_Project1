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
data = pd.read_csv("crimes_processed.csv")
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

random.seed()
seed = random.randint(1, 100)


X_train, X_test, y_train, y_test = train_test_split(
    data[cols], targetFitted, test_size=0.2, random_state=seed)
gnb.fit(X_train, y_train)
prediction = gnb.predict(X_test)

print(classification_report(y_test, prediction))
