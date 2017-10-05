# ITML_Project1

This is a project created for the Machine Learning course at Reykjavik University. The project was a supervised learning project.

## How to run

In order to run the training process and scoring for each model with its best chosen hyper-parameters run:

    ./run_tuned.sh

If you wish to run the hyper-parameter tuning code with cross validation you can run:

    ./run_crossvalidation_all.sh

Keep in mind that the the current code only cross validates on 5000 values for random forest and kNN. While it is 50000 for NaiveBayes. This is done to decrease the time it takes to do this step. In the accompanying report, these values were much higher.
