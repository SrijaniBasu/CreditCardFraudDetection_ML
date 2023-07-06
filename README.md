
# Credit Card Fraud Detection

This project aims to build a machine learning model for credit card fraud detection using a highly imbalanced dataset. The dataset contains transactions made by credit cards in September 2013 by European cardholders.

## Dataset Description

- Number of transactions: 284,807
- Number of fraud transactions: 492
- Percentage of fraud transactions: 0.173%
- Class imbalance ratio: 1 to 578.88
- Features: 'Time', 'V1', 'V2', ..., 'V28', 'Amount'
- Response variable: 'Class' (1 for fraud, 0 otherwise)

## Reference
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Approach

The project follows the following steps:

1. Loading the required libraries, including xgboost, pandas, numpy, scikit-learn, matplotlib, and TensorFlow.
2. Loading the dataset using `pd.read_csv` from the 'creditcard.csv' file.
3. Splitting the dataset into training and testing sets using `train_test_split`.
4. Performing feature scaling using `StandardScaler` to standardize the numerical features.
5. Training several models on the training set:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Deep Learning (Multi-Layer Perceptron)
   - Isolation Forest (Anomaly Detection)
6. Evaluating the models using Receiver Operating Characteristic (ROC) curves and calculating the Area Under the ROC Curve (AUC) for each model.
7. Plotting the ROC curves to compare the performance of the models.
8. Displaying the AUC values and the corresponding model names.
   
## Requirements

The following Python libraries are required to run this project:

- xgboost
- pandas
- numpy
- scikit-learn
- matplotlib
- TensorFlow

To install the required libraries, run the following command:

```bash
pip install xgboost pandas numpy scikit-learn matplotlib tensorflow


