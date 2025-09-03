import pandas as pd

# Load your data
df = pd.read_csv('dataset.csv')

# Convert 'Open Time' to datetime (if not already)
df['Open Time'] = pd.to_datetime(df['Open Time'])

# Sort by time (just in case)
df = df.sort_values('Open Time').reset_index(drop=True)

# Calculate the time difference between consecutive rows
df['Time Diff'] = df['Open Time'].diff()

# Filter rows where the time gap is not 5 minutes
gaps = df[df['Time Diff'] != pd.Timedelta(minutes=5)]

# Show the missing intervals
if gaps.empty:
    print("✅ No missing 5-minute intervals found.")
else:
    print(f"⚠️ Missing intervals detected: {len(gaps)}")
    print("Example gaps:")
    print(gaps[['Open Time', 'Time Diff']].head())

    print("\nMissing timestamps:")
    for i in range(len(gaps)):
        gap_index = gaps.index[i]
        if gap_index == 0:
            continue  # skip first row

        prev_time = df.loc[gap_index - 1, 'Open Time']
        current_time = df.loc[gap_index, 'Open Time']
        expected = prev_time + pd.Timedelta(minutes=5)
        while expected < current_time:
            print(expected)
            expected += pd.Timedelta(minutes=5)

