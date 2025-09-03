import pandas as pd
import numpy as np
import random
import json

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
    
def split_dataset(datapoints, train_ratio=0.7, test_ratio=0.2, eval_ratio=0.1, seed=42):
    # Shuffle the data
    random.seed(seed)
    random.shuffle(datapoints)

    # Compute split indices
    total = len(datapoints)
    train_end = int(train_ratio * total)
    test_end = train_end + int(test_ratio * total)

    # Split
    train = datapoints[:train_end]
    test = datapoints[train_end:test_end]
    evaluation = datapoints[test_end:]

    print(f"📦 Dataset split:")
    print(f"  🟢 Train: {len(train)}")
    print(f"  🔵 Test: {len(test)}")
    print(f"  🟡 Evaluation: {len(evaluation)}")

    return train, test, evaluation

def generate_datapoints(csv_path='dataset.csv', feature_window=512, label_window=256):
    # Load dataset
    df = pd.read_csv(csv_path)
    df['Open Time'] = pd.to_datetime(df['Open Time'])

    # Convert to NumPy arrays
    high = df['High'].to_numpy(dtype=float)
    low = df['Low'].to_numpy(dtype=float)
    open_price = df['Open'].to_numpy(dtype=float)
    open_time = df['Open Time'].to_numpy()

    # Generate datapoints
    total_rows = len(df)
    max_index = total_rows - (feature_window + label_window)
    datapoints = []

    for i in range(max_index):
        ft_start = i
        ft_end = i + feature_window
        lb_start = ft_end
        lb_end = lb_start + label_window

        open_time_temp=open_time[ft_end]
        open_price_temp=open_price[ft_end]

        high_features_temp=high[ft_start:ft_end]
        high_label_temp=np.max(high[lb_start:lb_end])
        high_mean_temp = high_features_temp.mean()
        high_std_temp = high_features_temp.std()

        high_features_temp = (high_features_temp - high_mean_temp) / high_std_temp
        high_label_temp = (high_label_temp - high_mean_temp) / high_std_temp

        low_features_temp=low[ft_start:ft_end]
        low_label_temp=np.min(low[lb_start:lb_end])
        low_mean_temp = low_features_temp.mean()
        low_std_temp = low_features_temp.std()

        low_features_temp = (low_features_temp - low_mean_temp) / low_std_temp
        low_label_temp = (low_label_temp - low_mean_temp) / low_std_temp

        dp = Datapoint(
            open_time = open_time_temp,
            open = open_price_temp,
            high_features = high_features_temp,
            high_label = high_label_temp,
            high_mean = high_mean_temp,
            high_std = high_std_temp,
            low_features = low_features_temp,
            low_label = low_label_temp,
            low_mean = low_mean_temp,
            low_std = low_std_temp,
        )

        datapoints.append(dp)

        # Print progress every 1000 steps
        if i % 1000 == 0 or i == max_index - 1:
            print(f"\rProgress: {i+1}/{max_index}", end='', flush=True)

    print(f"\n✅ Finished generating {len(datapoints)} Datapoint objects.")
    return datapoints

def save_high_npz(filename, datapoints):
    num_samples = len(datapoints)
    features = np.zeros((num_samples, 2, 512), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.float32)

    for i, dp in enumerate(datapoints):

        features[i, 0, :] = dp.high_features  # channel 0: normalized high_features
        features[i, 1, :] = dp.low_features   # channel 1: normalized low_features
        labels[i] = dp.high_label

        if i % 1000 == 0 or i == num_samples - 1:
            print(f"\rProcessing: {i+1}/{num_samples}", end='', flush=True)

    np.savez_compressed(
        filename,
        features=features,
        labels=labels
    )
    print(f"✅ Saved {filename}")

def save_low_npz(filename, datapoints):
    num_samples = len(datapoints)
    features = np.zeros((num_samples, 2, 512), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.float32)

    for i, dp in enumerate(datapoints):

        if i % 1000 == 0 or i == num_samples - 1:
            print(f"\rSaving: {i+1}/{num_samples}", end='', flush=True)

        features[i, 0, :] = dp.high_features  # channel 0: normalized high_features
        features[i, 1, :] = dp.low_features   # channel 1: normalized low_features
        labels[i] = dp.low_label

        if i % 1000 == 0 or i == num_samples - 1:
            print(f"\rProcessing: {i+1}/{num_samples}", end='', flush=True)

    np.savez_compressed(
        filename,
        features=features,
        labels=labels
    )
    print(f"✅ Saved {filename}")

def save_evaluation_json(filename, datapoints):
    # Save evaluation as-is (without normalization)
    json_list = []
    for dp in datapoints:
        json_list.append({
            "open_time": pd.to_datetime(dp.open_time).isoformat(),
            "open": dp.open,
            "high_features": dp.high_features.tolist(),
            "high_label": dp.high_label,
            "high_mean": dp.high_mean,
            "high_std": dp.high_std,
            "low_features": dp.low_features.tolist(),
            "low_label": dp.low_label,
            "low_mean": dp.low_mean,
            "low_std": dp.low_std
        })
    with open(filename, 'w') as f:
        json.dump(json_list, f, indent=2)
    print(f"✅ Saved {filename}")

if __name__ == "__main__":
    datapoints= generate_datapoints()
    train, test, evaluation = split_dataset(datapoints)

    # Save normalized npz files
    save_high_npz('/mnt/d/Binance/high_train_data.npz', train)
    save_low_npz('/mnt/d/Binance/low_train_data.npz', train)

    save_high_npz('/mnt/d/Binance/high_test_data.npz', test)
    save_low_npz('/mnt/d/Binance/low_test_data.npz', test)

    # Save evaluation raw json
    save_evaluation_json('/mnt/d/Binance/evaluation_data.json', evaluation)
