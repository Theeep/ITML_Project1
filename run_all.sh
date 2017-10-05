#!/bin/sh

echo "preprocessing starting"
python3 data_preprocessing.py

echo "preprocessing done"
cp crimes_processed.csv kNN
cp crimes_processed.csv naive_bayes
cp crimes_processed.csv random_forest

echo "preprocessing kNN starting" 
python3 kNN/data_preprocessing_kNN.py
echo "preprocessing kNN done"
echo "Training kNN starting"
python3 kNN/kNN_tuned_hyperparameters.py
echo "Training kNN done"
echo "Training random forest starting"
python3 random_forest/random_forest_with_hyperparam.py
echo "Training random forest done"
echo "Training naive bayes starting"
python3 naive_bayes/naive_bayes_with_hyperparam.py
echo "Training naive bayes done"
