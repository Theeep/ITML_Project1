import pandas as pd
import sklearn.pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif

crime_data = pd.read_csv("crimesProcessedkNN.csv", nrows=150000)

crime_labels = crime_data["TYPE"]
crime_data = crime_data.drop("TYPE", axis=1)

knn = KNeighborsClassifier(n_jobs=4)
feature_selector = SelectKBest(f_classif)

steps = [
    ('feature_selection', feature_selector),
    ('knn_classifier', knn)
]

parameters = dict(
    feature_selection__k=[10, 20, 30],
    knn_classifier__n_neighbors=[5, 10, 15],
    knn_classifier__leaf_size=[20,30],
    knn_classifier__algorithm=['ball_tree', 'kd_tree'],
    knn_classifier__weights=['uniform', 'distance'],
    knn_classifier__metric=['chebyshev', 'minkowski']

)

for i in [(RobustScaler(), "RobustScaler"), (MinMaxScaler(), "MinMaxScaler")]:

    crime_data_scaled = i[0].fit_transform(crime_data)
    X_train, X_test, y_train, y_test = train_test_split(crime_data_scaled, crime_labels, 
                                                    test_size=0.20)

    knn_pipeline = sklearn.pipeline.Pipeline(steps)
    grid_search = GridSearchCV(knn_pipeline, param_grid=parameters, n_jobs=4)

    grid_search.fit(X_train, y_train)
    y_prediction = grid_search.predict(X_train)

    report = classification_report(y_train, y_prediction)
    print(report)
    print(grid_search.best_estimator_.get_params()["steps"])

    result_df = pd.DataFrame(grid_search.cv_results_)
    print(result_df)
    result_df.to_csv(i[1]+"Results.csv", encoding='utf-8', index=False)