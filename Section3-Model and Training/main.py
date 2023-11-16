import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from scipy.stats import skew, kurtosis
import seaborn as sns
from log_reg import MyLogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import argparse
import json
import joblib
import yaml
import sys

def training(lr, iteration, batch_size, file, features):
    data_raw = pd.read_csv(file)
    data = data_raw.dropna()
    scaler = StandardScaler()

    X_raw = data[features]
    y = data[["Hogwarts House"]]
    X = scaler.fit_transform(X_raw)

    #Step 1: clean up the data
    logistic_regression = MyLogisticRegression(alpha=lr, max_iter=iteration, batch=batch_size)
    X_ = logistic_regression.add_intercept(X)

    # Step 2: Split the data 
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=42)

    # Step 3: Fit the model, visualize the training progress and saves the model
    logistic_regression.fit(X_train, y_train)

    # Step 4: Predict the class labels for your test data
    y_pred = logistic_regression.predict(X_test)

    # print("validation prediction: ", y_pred)
    confusion_mat = logistic_regression.confusion_matrix(y_test.squeeze(), y_pred)

    labels = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]
    performance = logistic_regression.performance(y_test.squeeze(), y_pred, labels)
    
    print(f"Category/performance  : precision  recall  accuracy f1_score")
    print(f"{labels[0]}:            {(performance['precision'][0]): 0.2f}    {(performance['recall'][0]): 0.2f}  {(performance['accuracy'][0]): 0.2f}  {(performance['f1_score'][0]): 0.2f}")
    print(f"{labels[1]}:             {(performance['precision'][1]): 0.2f}    {(performance['recall'][1]): 0.2f}  {(performance['accuracy'][1]): 0.2f}  {(performance['f1_score'][1]): 0.2f}")
    print(f"{labels[2]}:             {(performance['precision'][2]): 0.2f}    {(performance['recall'][2]): 0.2f}  {(performance['accuracy'][2]): 0.2f}  {(performance['f1_score'][2]): 0.2f}")
    print(f"{labels[3]}:            {(performance['precision'][3]): 0.2f}    {(performance['recall'][3]): 0.2f}  {(performance['accuracy'][3]): 0.2f}  {(performance['f1_score'][3]): 0.2f}")


def testing(file, model_file):
    data = pd.read_csv(file)
    scaler = StandardScaler()

    X_raw = data[["Herbology", "Astronomy", "Ancient Runes", "Defense Against the Dark Arts"]]
    X = scaler.fit_transform(X_raw)

    logistic_regression = MyLogisticRegression()
    logistic_regression.load_model(model_file)

    X_ = logistic_regression.add_intercept(X)

    y_pred = logistic_regression.predict(X_)

    with open('result.json', 'w') as json_file:
        json.dump(y_pred.tolist(), json_file)
    
if __name__ == "__main__":
    
    with open(sys.argv[1], 'r') as file:
        config = yaml.safe_load(file)
    
    lr = config['lr']
    epoches = config['epoches']
    batch_size = config['batch_size']
    file = config['data_file']
    model = config['model_file']
    mode = config['mode']
    features = [key for key, value in config['features'].items() if value == True]

    if (mode == "train"):
        training(lr, epoches, batch_size, file, features)
    elif (mode == "test"):
        if model is None:
            print("model file is not provided")
        testing(file, model)
    else:
        print("Mode is not provided")