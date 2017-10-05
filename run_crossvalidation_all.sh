#!/bin/sh

echo "Preprocessing starting"
python3 data_preprocessing.py
cp crimes_processed.csv kNN
cp crimes_processed.csv naive_bayes
cp crimes_processed.csv random_forest
echo "Preprocessing done"

echo "Preprocessing kNN starting"
python3 kNN/data_preprocessing_kNN.py
cp crimes_processed_kNN.csv kNN
echo "Preprocessing kNN done"

echo "Cross validation kNN starting"
python3 kNN/kNN_tuned_hyperparameters.py
echo "Cross validation kNN Done see results in kNN/robust_scaler_results.csv"

echo "Cross validation random forest starting"
python3 random_forest/random_forest.py
echo "Cross validation random forest done see results in random_forest/RandomForestResults.csv"

echo "Cross validation naive bayes starting"
python3 naive_bayes/NaiveBayes.py
echo "Cross validation naive bayes done see results in naive_bayes/naive_results.csv"
