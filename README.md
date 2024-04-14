# ML-Interview-Task-Solution

## Description
This repository contains the solution to the machine learning task for the prediction of subjects of scientific papers. The entire dataset is available in this repository in the dataset folder. However, for this particular task, only the cora.content file was necessary.

## Installation Requirements
The packages used in this project are available in "requirements.txt" file. If they are not already installed, it could be installed by the following command <br />
"`pip install -r requirements.txt`"

## Execution
The parameters necessary for the execution are available in the config.yaml file in configs folder. A separate configuration file has been used to separate the code from the data. No parameters are hardcorded in the scripts and hence different experiments can be run by just changing the parameter values in the configuration file. The approach could be executed by the following command <br />
**`python main.py -c configs/config.yaml`** <br />
The predictions of this approach is stored in the file "predictions.tsv"

## Approach
Only the cora.content files was loaded because the links between the papers were not necessary for this task. The dataset was split into train set and test set with a test size of 0.2 in order to prevent overfitting and evaluate the performance of the model on samples not seen by the model during training. A 10-fold cross validation was performed on the train set and the cross_val_score for each of these folds as validation set has been shown during the course of execution of the task. Because of the high dimensionality of the dataset (1433 dimensional feature vector for each sample), a `Support Vector Classifier` with `Radial Basis Function(RBF)` kernel was deemed fit for the purpose. The predictions generated for the entire dataset(both the train and test set) has been stored in the "predictions.tsv" files as `<paper_id> <class_label>`, where "paper_id" is a unique ID for each paper in the dataset, and "class_label" is one of seven classes (`Case_Based`, `Genetic_Algorithms`, `Neural_Networks`, `Probabilistic_Methods`, `Reinforcement_Learning`, `Rule_Learning`, `Theory`), the scientific papers are classified into. The accuracy of the model for both the train set and the test set has been shown along the course of execution of the program. A training accuracy of 95.66% and test accuracy of 73.99% was achieved. A seed value was used throughtout the program to facilitate reproducibility of the results. 