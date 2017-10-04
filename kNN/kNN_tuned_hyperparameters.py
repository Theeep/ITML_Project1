import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif

crime_data = pd.read_csv("crimes_processed_kNN.csv")
crime_labels = crime_data["TYPE"]
crime_data = crime_data.drop("TYPE", axis=1)
# unique_types = crime_data.TYPE.unique()
# print(unique_types)
# sns.countplot(x=crime_data.TYPE, order=unique_types)
# plt.show()

scaler = RobustScaler()
crime_data_scaled = scaler.fit_transform(crime_data)
X_train, X_test, y_train, y_test = train_test_split(crime_data_scaled, crime_labels, 
                                                    test_size=0.20)

feature_selector = SelectKBest(f_classif, k=20)
X_train = feature_selector.fit_transform(X_train, y_train)
X_test = feature_selector.transform(X_test)

knn = KNeighborsClassifier(n_jobs=4, algorithm='ball_tree', leaf_size=20, metric='minkowski', n_neighbors=15, weights='uniform')
knn.fit(X_train,y_train)

print("Training")
y_prediction = knn.predict(X_train)
report = classification_report(y_train, y_prediction)
print(report)
print(knn.score(X_train, y_train))

print("")

print("Test")
y_prediction = knn.predict(X_test)
report = classification_report(y_test, y_prediction)
print(report)
print(knn.score(X_test, y_test))