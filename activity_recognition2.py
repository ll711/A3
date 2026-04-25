import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
import itertools

from train_model import five_fold_cross_validation, evaluate_generalized_model

# Index for each activity
activity_indices = {
    'Stationary': 0,
    'Walking-flat-surface': 1,
    'Walking-up-stairs': 2,
    'Walking-down-stairs': 3,
    'Elevator-up': 4,
    'Running': 5,
    'Elevator-down': 6
}


def compute_raw_data(dir_name):
    """Load raw sensor data, interpolate, and smooth barometer."""
    raw_data_features = None
    raw_data_labels = None
    interpolated_timestamps = None

    sessions = set()
    file_dict = dict()
    file_names = os.listdir(dir_name)

    for file_name in file_names:
        if '.txt' in file_name:
            tokens = file_name.split('-')
            if len(tokens) < 6:
                continue
            identifier = '-'.join(tokens[4:6])
            activity = '-'.join(file_name.split('-')[6:-2])
            sensor = tokens[-1]
            sessions.add((identifier, activity))

            if (identifier, activity, sensor) in file_dict:
                file_dict[(identifier, activity, sensor)].append(file_name)
            else:
                file_dict[(identifier, activity, sensor)] = [file_name]

    for session in sessions:
        accel_file = file_dict[(session[0], session[1], 'accel.txt')][0]
        accel_df = pd.read_csv(dir_name + '/' + accel_file)
        accel = accel_df.drop_duplicates(accel_df.columns[0], keep='first').values
        timestamps = np.arange(accel[0, 0] + 3000.0, accel[-1, 0] - 3000.0, 1000.0 / 32)

        accel = np.stack([np.interp(timestamps, accel[:, 0], accel[:, 1]),
                          np.interp(timestamps, accel[:, 0], accel[:, 2]),
                          np.interp(timestamps, accel[:, 0], accel[:, 3])],
                         axis=1)

        bar_file = file_dict[(session[0], session[1], 'pressure.txt')][0]
        bar_df = pd.read_csv(dir_name + '/' + bar_file)
        bar = bar_df.drop_duplicates(bar_df.columns[0], keep='first').values
        bar = np.interp(timestamps, bar[:, 0], bar[:, 1]).reshape(-1, 1)
        bar = sm.nonparametric.lowess(bar[:, 0], timestamps, return_sorted=False).reshape(-1, 1)

        length_multiple_128 = 128 * int(bar.shape[0] / 128)
        accel = accel[0:length_multiple_128, :]
        bar = bar[0:length_multiple_128, :]
        labels = np.array(bar.shape[0] * [int(activity_indices[session[1]])]).reshape(-1, 1)
        timestamps = timestamps[0:length_multiple_128]

        if raw_data_features is None:
            raw_data_features = np.append(accel, bar, axis=1)
            raw_data_labels = labels
            interpolated_timestamps = timestamps
        else:
            raw_data_features = np.append(raw_data_features, np.append(accel, bar, axis=1), axis=0)
            raw_data_labels = np.append(raw_data_labels, labels, axis=0)
            interpolated_timestamps = np.append(interpolated_timestamps, timestamps, axis=0)

    return raw_data_features, raw_data_labels, interpolated_timestamps


def plot_raw_data(raw_data_features, raw_data_labels):
    """Plot raw accelerometer magnitude, barometric pressure and activity labels."""
    accel_magnitudes = np.sqrt((raw_data_features[:, 0] ** 2).reshape(-1, 1) +
                               (raw_data_features[:, 1] ** 2).reshape(-1, 1) +
                               (raw_data_features[:, 2] ** 2).reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(accel_magnitudes)
    plt.xticks(fontsize=8)
    plt.ylabel('Acceleration (m/s^2)', fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().set_title('Accelerometer Magnitude', fontsize=8)

    plt.subplot(3, 1, 2)
    plt.plot(raw_data_features[:, 3])
    plt.xticks(fontsize=8)
    plt.ylabel('Pressure (mbar)', fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().set_title('Barometric Pressure', fontsize=8)

    plt.subplot(3, 1, 3)
    plt.plot(raw_data_labels)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('Activity', fontsize=8)
    plt.gca().set_title('Activity Label', fontsize=8)
    plt.grid(True)

    os.makedirs('./data_process', exist_ok=True)
    plt.savefig('./data_process/raw_data.png', bbox_inches='tight')
    plt.show()


def feature_extraction(raw_data_features, raw_data_labels, timestamps):
    """Extract features from raw sensor data using sliding windows."""
    features = None
    labels = None

    accel_magnitudes = np.sqrt((raw_data_features[:, 0] ** 2).reshape(-1, 1) +
                               (raw_data_features[:, 1] ** 2).reshape(-1, 1) +
                               (raw_data_features[:, 2] ** 2).reshape(-1, 1))

    segment_size = 128

    for i in range(0, accel_magnitudes.shape[0] - segment_size, 64):
        segment = accel_magnitudes[i:i + segment_size, :]
        if len(segment) == 0:
            continue

        accel_mean = np.mean(segment)
        accel_var = np.var(segment)

        segment_fft_powers = np.abs(np.fft.fft(segment)) ** 2

        equal_band_power = []
        window_size = 32
        for j in range(0, len(segment_fft_powers), window_size):
            equal_band_power.append(sum(segment_fft_powers[j:j + 32]).tolist()[0])

        log_band_power = []
        freqs = [0, 2, 4, 8, 16, 32, 64, 128]
        for j in range(len(freqs) - 1):
            log_band_power.append(sum(segment_fft_powers[freqs[j]:freqs[j + 1]]).tolist()[0])

        bar_slope = np.polyfit(timestamps[i:i + segment_size],
                               raw_data_features[i:i + segment_size, 3], 1)[0]

        feature = [accel_mean, accel_var] + equal_band_power + log_band_power + [bar_slope]

        if features is None:
            features = np.array([feature])
        else:
            features = np.append(features, [feature], axis=0)

        label = Counter(raw_data_labels[i:i + segment_size][:, 0].tolist()).most_common(1)[0][0]

        if labels is None:
            labels = np.array([label])
        else:
            labels = np.append(labels, [label], axis=0)

    return features, labels


def plot_extracted_features(features, labels):
    """Plot all extracted features in one figure (To Do 2)."""
    plt.figure(figsize=(12, 24))

    plt.subplot(7, 1, 1)
    plt.plot(features[:, 0])
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.ylabel('m/s^2', fontsize=8)
    plt.gca().set_title('Mean of Accelerometer Magnitude', fontsize=8)

    plt.subplot(7, 1, 2)
    plt.plot(features[:, 1])
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.ylabel('m/s^2', fontsize=8)
    plt.gca().set_title('Variance of Accelerometer Magnitude', fontsize=8)

    plt.subplot(7, 1, 3)
    plt.plot(features[:, 2])
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.ylabel('Power', fontsize=8)
    plt.gca().set_title('FFT Equispaced Band 1', fontsize=8)

    plt.subplot(7, 1, 4)
    plt.plot(features[:, 3])
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.ylabel('Power', fontsize=8)
    plt.gca().set_title('FFT Equispaced Band 2', fontsize=8)

    plt.subplot(7, 1, 5)
    plt.plot(features[:, 6])
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.ylabel('Power', fontsize=8)
    plt.gca().set_title('FFT Log Band 1', fontsize=8)

    plt.subplot(7, 1, 6)
    plt.plot(features[:, -1])
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.ylabel('mbar/s', fontsize=8)
    plt.gca().set_title('Slope of Barometric Pressure', fontsize=8)

    plt.subplot(7, 1, 7)
    plt.plot(labels)
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.gca().set_title('Activity', fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    os.makedirs('./data_process', exist_ok=True)
    plt.savefig('./data_process/extracted_features_comprehensive.png', bbox_inches='tight')
    plt.show()


def plot_feature_boxplots(features, labels, filename='feature_boxplots.png'):
    """Boxplots of key features grouped by activity (To Do 6)."""
    idx_to_activity = {v: k for k, v in activity_indices.items()}
    unique_labels = np.unique(labels)
    activity_names = [idx_to_activity[int(lbl)] for lbl in unique_labels]

    accel_vars = [features[labels.flatten() == lbl, 1] for lbl in unique_labels]
    baro_slopes = [features[labels.flatten() == lbl, -1] for lbl in unique_labels]

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.boxplot(accel_vars, tick_labels=activity_names)
    plt.ylabel('Variance (m/s^2)', fontsize=10)
    plt.title('Acceleration Variance across Activities', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.boxplot(baro_slopes, tick_labels=activity_names)
    plt.ylabel('Slope (mbar/s)', fontsize=10)
    plt.title('Barometric Pressure Slope across Activities', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs('./data_process', exist_ok=True)
    plt.savefig(f'./data_process/{filename}', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    data_path = './uploaded/'
    my_netid = 'cuu25pbu'   # Change to your own NetID

    raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + my_netid)

    "***Plot Raw Data***"
    plot_raw_data(raw_data_features, raw_data_labels)

    "***Feature Extraction***"
    features, labels = feature_extraction(raw_data_features, raw_data_labels, timestamps)

    "***Plot Features***"
    plot_extracted_features(features, labels)

    "***Classify User's Own Data (Within-subject, no extra data)***"
    five_fold_cross_validation(features, labels, activity_indices)

    "***Person-independent model (Between-subjects, with extra data)***"
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None

    dirs = os.listdir(data_path)
    for dir in dirs:
        print("Processing data from %s" % dir)
        raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + dir)
        features, labels = feature_extraction(raw_data_features, raw_data_labels, timestamps)

        if dir == my_netid:
            X_test = features
            Y_test = labels
        else:
            if X_train is None:
                X_train = features
                Y_train = labels
            else:
                X_train = np.append(X_train, features, axis=0)
                Y_train = np.append(Y_train, labels, axis=0)

    if X_train is not None and Y_train is not None:
        evaluate_generalized_model(X_train, Y_train, X_test, Y_test, activity_indices, plot_feature_boxplots)
    else:
        print(
            "No training data from other subjects found. Skipping person-independent model evaluation. "
            "Please add other users' data to the 'uploaded' directory to train the model.")