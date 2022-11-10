# churn-predict

Telco Customer Churn Prediction

Assignment 1 for Data Analytics Fundamentals &
Assignment 2 for Data Analytics Fundamentals.

## Dataset

The data set used is from Kaggle Customer Churn Prediction 2020 competition.
https://www.kaggle.com/competitions/customer-churn-prediction-2020/data

## Approaches

The following ML Approaches are tried out to predict

1. K- Nearest Neighbours
2. Logistic Regression
3. Random Forest
4. Neural Network

## Hyper Parameter Optimization

HPO is done for the neural network approach.
HPO is done using Optuna Library.

The following parameters are tuned. 
1. Learning Rate
2. Batch Size
3. Number of layers
4. Number of Neurons in Layer
5. Dropout 
6. Activation Function

## Repo Structure

```bash
├── data - "Directory for storing data"
├── models - "Directory to store models during and after training"
├── scripts
│   ├── knn-script.py - "K nearest neighbour algorithm execution script"
│   ├── logistic-regression-script.py - "Logistic Regression algorithm execution script"
│   ├── neural-network-hpo-script.py - "Neural network hyper parameter optimization script"
│   ├── neural-network-script.py - "Neural network execution script"
│   └── random-forrest-script.py - "Random Forest algorithm execution script"
├── src
│   ├── __init__.py
│   ├── constants.py
│   ├── nn_utils.py - "Tunable NN model class"
│   ├── nn_utils.py - "NN model class, Dataset class, and training utilities"
│   └── utils.py - "Data preprocessing and evaluation utilities"
├── LICENSE
├── README.md
└── requirements.txt
```

The scripts should be run from the root directory (instead of the scripts directory)
