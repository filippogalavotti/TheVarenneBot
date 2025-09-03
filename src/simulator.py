import numpy as np
import pandas as pd
import tensorflow as tf
import json
from datetime import datetime
import random
from scipy.optimize import minimize

class Datapoint:
    def __init__(self, open_time, open, high_features, high_label, high_mean, high_std, low_features, low_label, low_mean, low_std):
        self.open_time = open_time
        self.open = open
        self.high_features = high_features
        self.high_label = high_label
        self.high_mean = high_mean
        self.high_std = high_std
        self.low_features = low_features
        self.low_label = low_label
        self.low_mean = low_mean
        self.low_std = low_std

    def __repr__(self):
        return (
            f"Datapoint(Open Time: {self.open_time}, Open: {self.open}, "
            f"High Label: {self.high_label}, Low Label: {self.low_label})"
        )

def load_evaluation_json(filename):
    with open(filename, 'r') as f:
        json_list = json.load(f)

    datapoints = []
    for item in json_list:
        dp = Datapoint(
            open_time=datetime.fromisoformat(item['open_time']),
            open=item['open'],
            high_features=np.array(item['high_features']),
            high_label=item['high_label'],
            high_mean=item['high_mean'],
            high_std=item['high_std'],
            low_features=np.array(item['low_features']),
            low_label=item['low_label'],
            low_mean=item['low_mean'],
            low_std=item['low_std']
        )
        datapoints.append(dp)

    print(f"✅ Loaded {filename} with {len(datapoints)} datapoints")
    return datapoints

def simRun(HIGHmodel, LOWmodel, evaluation_data, balance, stake):

    iterativeSum = 0

    step = int(512*(stake / balance))

    for j in range(step):

        print(f"STARTING ITERATION N{j+1}")

        for i, dp in enumerate(evaluation_data):

            if (i % step) == j:

                print(f"ITERATION N{j+1} - ELABORATING: {i} / {len(evaluation_data)}")

                diff = 0

                features = np.zeros((1, 512, 2), dtype=np.float32)

                features[0, :, 0] = dp.high_features  # channel 0: normalized high_features
                features[0, :, 1] = dp.low_features   # channel 1: normalized low_features

                high = HIGHmodel.predict(features)[0][0]

                high = ( high * dp.high_std ) + dp.high_mean

                low = LOWmodel.predict(features)[0][0]

                low = ( low * dp.low_std ) + dp.low_mean

                high_label = ( dp.high_label * dp.high_std ) + dp.high_mean

                low_label = ( dp.low_label * dp.low_std ) + dp.low_mean

                print(f"Time: {dp.open_time} - Balance : {balance:.2f} $ - Open value: {dp.open}")

                print(f"High prediction: {high:.2f} ({high_label:.2f}), Low prediction: {low:.2f} ({low_label:.2f})")

                if ((high <= low) or (dp.open > high) or (dp.open < low) or (balance < stake)):
                    print("NO POSITION")
                    continue

                high_delta = high - dp.open
                low_delta = dp.open - low

                if (high_delta > low_delta):  #Long position
                
                    if (low_label <= low) or (high_label < high):

                        diff = stake * (((low + random.randint(-100, 100)) / (dp.open + random.randint(-100, 100))) - 1) - 0.01

                        print(f"LONG POSITION : {diff:.2f} $ (LOST)")

                        balance += diff

                    else:

                        diff = stake * (((high + random.randint(-100, 100)) / (dp.open + random.randint(-100, 100))) - 1) - 0.01

                        print(f"LONG POSITION : {diff:.2f} $ (WON)")

                        balance += diff

                else: # Short Position

                    if (high_label >= high) or (low_label > low):

                        # diff = balance * stake * (low - dp.open) / dp.open

                        diff = stake * (1 - ((high + random.randint(-100, 100)) / (dp.open + random.randint(-100, 100)))) - 0.01

                        print(f"SHORT POSITION : {diff:.2f} $ (LOST)")

                        balance += diff

                    else:

                        diff = stake * (1 - ((low + random.randint(-100, 100)) / (dp.open + random.randint(-100, 100)))) - 0.01

                        print(f"SHORT POSITION : {diff:.2f} $ (WON)")

                        balance += diff

        print(f"FINAL BALANCE : {balance} $")

        iterativeSum += balance
    
    averageReturn = (iterativeSum / (step * balance)) - 1

    print(f"AVERAGE RETURN: {averageReturn}")

    return (averageReturn)

if __name__ == "__main__":

    # IMPORT MODELS

    HIGHmodel = tf.keras.models.load_model('/root/Data/Varenne/HIGHModel.keras') # Created by HIGHModel.py
    LOWmodel = tf.keras.models.load_model('/root/Data/Varenne/LOWModel.keras') # Created by LOWModel.py

    # IMPORT DATA

    evaluation_data = load_evaluation_json("/mnt/d/Binance/evaluation_data.json") # Created by dataset_generator.py
    print(evaluation_data[0])
    print(evaluation_data[0].high_features.shape)

    def objective(x):
        balance, stake = x

        return -simRun(HIGHmodel, LOWmodel, evaluation_data, balance, stake)

    initial_guess = [1000.0, 50.0]
    bounds = [(0, 7000.0), (0, 500.0)]

    result = minimize(
        objective,
        initial_guess,
        bounds=bounds,
        method='Powell',
        options={
            'maxiter': 10,
            'disp': True
        })

    print(f"\nOptimized balance and stake: {result.x}")
    print(f"Maximum simRun value: {result.fun}")





    
