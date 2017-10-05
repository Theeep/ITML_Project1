# ITML_Project1

This is a project created for the Machine Learning course at Reykjavik University. The project was a superwised learning project.

## How to run

In order to run a pipeline which runs each factor of the project not including the CrossValidation run the bash script run_all.sh

    ./run_all.sh

If you wish to run the cross validation you can run run_crossvalidation_all.sh

    ./run_crossvalidation_all.sh

Keep in mind that the the current code only cross validates on 5000 values for random forest and kNN. While it is 50000 for NaiveBayes. This is done to decrease the time it takes to do this step.
