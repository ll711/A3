import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
import itertools


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

  raw_data_features = None
  raw_data_labels = None
  interpolated_timestamps = None

  sessions = set()
  # Categorize files containing different sensor sensor data
  file_dict = dict()
  # List of different activity names
  activities = set()
  file_names = os.listdir(dir_name)

  for file_name in file_names:
    if '.txt' in file_name:
      tokens = file_name.split('-')
      if len(tokens) < 6:
          continue
      identifier = '-'.join(tokens[4: 6])
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
    # Spine-line interpolataion for x, y, z values (sampling rate is 32Hz).
    # Remove data in the first and last 3 seconds.
    timestamps = np.arange(accel[0, 0]+3000.0, accel[-1, 0]-3000.0, 1000.0/32)

    accel = np.stack([np.interp(timestamps, accel[:, 0], accel[:, 1]),
                      np.interp(timestamps, accel[:, 0], accel[:, 2]),
                      np.interp(timestamps, accel[:, 0], accel[:, 3])],
                     axis=1)

    bar_file = file_dict[(session[0], session[1], 'pressure.txt')][0]
    bar_df = pd.read_csv(dir_name + '/' + bar_file)
    bar = bar_df.drop_duplicates(bar_df.columns[0], keep='first').values
    bar = np.interp(timestamps, bar[:, 0], bar[:, 1]).reshape(-1, 1)

    # Apply lowess to smooth the barometer data with window-size 128
    # bar = np.convolve(bar[:, 0], np.ones(128)/128, mode='same').reshape(-1, 1)
    bar = sm.nonparametric.lowess(bar[:, 0], timestamps, return_sorted=False).reshape(-1, 1)

    # Keep data with dimension multiple of 128
    length_multiple_128 = 128*int(bar.shape[0]/128)
    accel = accel[0:length_multiple_128, :]
    bar = bar[0:length_multiple_128, :]
    labels = np.array(bar.shape[0]*[int(activity_indices[session[1]])]).reshape(-1, 1)
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
  """ This function plots the raw data features (after applying basic data processing) and raw data labels.
      The first subplot is the the accelerometer magnitude. The second subplot is the barometric pressure.
      The third subplot is the activity label (check "activity_indices" to see what activity each index corresponds to).
  """
  accel_magnitudes = np.sqrt((raw_data_features[:, 0]**2).reshape(-1, 1)+
                             (raw_data_features[:, 1]**2).reshape(-1, 1)+
                             (raw_data_features[:, 2]**2).reshape(-1, 1))

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

  os.makedirs('./data_processing', exist_ok=True)
  plt.savefig('./data_processing/raw_data.png', bbox_inches='tight')
  plt.show()


def feature_extraction(raw_data_features, raw_data_labels, timestamps):
  """    raw_data_features: The fourth column is the barometer data.

  Returns:
    features: Features extracted from the data features, where
              features[:, 0:3] is the slope of 3-axis acceleration (x, y, z);
              features[:, 3:7] is the fft power spectrum of equally-spaced frequencies;
              features[:, 7] is the slope of pressure.
  Args:

  """
  features = None
  labels = None

  accel_magnitudes = np.sqrt((raw_data_features[:, 0]**2).reshape(-1, 1)+
                             (raw_data_features[:, 1]**2).reshape(-1, 1)+
                             (raw_data_features[:, 2]**2).reshape(-1, 1))

  # The window size for feature extraction
  segment_size = 128


  for i in range(0, accel_magnitudes.shape[0]-segment_size, 64):
    segment = accel_magnitudes[i:i+segment_size, :]

    if len(segment) == 0:
        continue

    # 3-axis acceleration slopes
    t_segment = timestamps[i:i+segment_size]
    accel_x_slope = np.polyfit(t_segment, raw_data_features[i:i+segment_size, 0], 1)[0]
    accel_y_slope = np.polyfit(t_segment, raw_data_features[i:i+segment_size, 1], 1)[0]
    accel_z_slope = np.polyfit(t_segment, raw_data_features[i:i+segment_size, 2], 1)[0]

    segment_fft_powers = np.abs(np.fft.fft(segment, axis=0))**2

    # Band power of equally-spaced bands
    equal_band_power = list()
    window_size = 32
    for j in range(0, len(segment_fft_powers), window_size):
      equal_band_power.append(sum(segment_fft_powers[j: j+32]).tolist()[0])

    # Slope of barometer data
    bar_slope = np.polyfit(t_segment, raw_data_features[i:i+segment_size, 3], 1)[0]

    # Combine the three types of features requested
    feature = [accel_x_slope, accel_y_slope, accel_z_slope] + equal_band_power + [bar_slope]

    if features is None:
      features = np.array([feature])
    else:
      features = np.append(features, [feature], axis=0)

    label = Counter(raw_data_labels[i:i+segment_size][:, 0].tolist()).most_common(1)[0][0]

    if labels is None:
      labels = np.array([label])
    else:
      labels = np.append(labels, [label], axis=0)

  return features, labels


def plot_extracted_features(features, labels):
  """ This function plots the extracted features. The top plot is the slope of accelerometer X-axis data.
      The middle plot is the slope of barometric pressure data. The bottom plot is the activity label.
  """
  # Plot the acceleration X-axis slope
  plt.subplot(3, 1, 1)
  plt.plot(features[:, 0])
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.ylabel('Slope', fontsize=8)
  plt.gca().set_title('Slope of Accelerometer X-axis', fontsize=8)

  # Plot the barometer slope
  plt.subplot(3, 1, 2)
  plt.plot(features[:, -1])
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.ylabel('mbar/s', fontsize=8)
  plt.gca().set_title('Slope of Barometric Pressure', fontsize=8)

  plt.subplot(3, 1, 3)
  plt.plot(labels)
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.gca().set_title('Activity', fontsize=8)
  plt.grid(True)

  os.makedirs('./data_process', exist_ok=True)
  plt.savefig('./data_process/extracted_features.png', bbox_inches='tight')
  plt.show()


class HierarchicalClassifier:
    def __init__(self):
        self.clf_level1 = DecisionTreeClassifier()
        self.clf_level2_slope = DecisionTreeClassifier()
        self.clf_level2_fft = DecisionTreeClassifier()

    def fit(self, X, y):
        # Level 1: is it (Elevator/Stairs) or (Running/Stationary/Walking-flat)
        # Groups: Elevator/Stairs = 1 (Slope preferred), Running/Stationary/Flat = 0 (FFT preferred)
        y_level1 = np.isin(y, [2, 3, 4, 6]).astype(int)
        self.clf_level1.fit(X, y_level1)

        # Train Level 2 - Slope
        mask_slope = y_level1 == 1
        mask_fft = y_level1 == 0
        if np.any(mask_slope):
            # Emphasis on slopes (idx 0,1,2,7)
            weights_slope = np.ones(X.shape[1])
            weights_slope[[0,1,2,7]] = 10.0
            self.clf_level2_slope.fit(X[mask_slope] * weights_slope, y[mask_slope])
        if np.any(mask_fft):
            # Emphasis on FFT (idx 3,4,5,6)
            weights_fft = np.ones(X.shape[1])
            weights_fft[[3,4,5,6]] = 10.0
            self.clf_level2_fft.fit(X[mask_fft] * weights_fft, y[mask_fft])
        return self

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        l1_preds = self.clf_level1.predict(X)

        mask_slope = l1_preds == 1
        mask_fft = l1_preds == 0

        if np.any(mask_slope):
            weights_slope = np.ones(X.shape[1])
            weights_slope[[0,1,2,7]] = 10.0
            preds[mask_slope] = self.clf_level2_slope.predict(X[mask_slope] * weights_slope)

        if np.any(mask_fft):
            weights_fft = np.ones(X.shape[1])
            weights_fft[[3,4,5,6]] = 10.0
            preds[mask_fft] = self.clf_level2_fft.predict(X[mask_fft] * weights_fft)
        return preds


def five_fold_cross_validation(features, labels):

  true_labels = list()
  predicted_labels = list()

  for train_index, test_index in StratifiedKFold(n_splits=5).split(features, labels):
    X_train = features[train_index, :]
    Y_train = labels[train_index]

    X_test = features[test_index, :]
    Y_test = labels[test_index]

    clf = HierarchicalClassifier()
    clf.fit(X_train, Y_train.flatten())
    predicted_label = clf.predict(X_test)

    predicted_labels += predicted_label.flatten().tolist()
    true_labels += Y_test.flatten().tolist()

  # Given N different activities, the confusion matrix is a N*N matrix
  confusion_matrix = np.zeros((len(activity_indices), len(activity_indices)))

  for i in range(len(true_labels)):
    confusion_matrix[int(true_labels[i]), int(predicted_labels[i])] += 1

  # Normalized confusion matrix
  #for i in range(confusion_matrix.shape[0]):
   # print(sum(confusion_matrix[i, :]))
   #  confusion_matrix[i, :] = confusion_matrix[i, :]/sum(confusion_matrix[i, :])

  print("===== Confusion Matrix (within subject) =====")
  plot_confusion_matrix(confusion_matrix, activity_indices.keys(), normalize=False)
  os.makedirs('./data_processing', exist_ok=True)
  plt.savefig('./data_processing/confusion_matrix_within_subject.png', bbox_inches='tight')
  plt.show()


def evaluate_generalized_model(X_train, Y_train, X_test, Y_test):
  clf = HierarchicalClassifier().fit(X_train, Y_train.flatten())
  Y_pred = clf.predict(X_test)

  # Given N activities, the confusion matrix is a N*N matrix
  confusion_matrix = np.zeros((len(activity_indices), len(activity_indices)))

  for i in range(len(Y_test)):
    confusion_matrix[int(Y_test[i]), int(Y_pred[i])] += 1

  # print(confusion_matrix)

  # for i in range(confusion_matrix.shape[0]):
  #   # print(sum(confusion_matrix[i, :]))
  #   confusion_matrix[i, :] = confusion_matrix[i, :]/sum(confusion_matrix[i, :])

  print("===== Confusion Matrix (between subjects) =====")
  plot_confusion_matrix(confusion_matrix, activity_indices.keys())
  os.makedirs('./data_processing', exist_ok=True)
  plt.savefig('./data_processing/confusion_matrix_between_subjects.png', bbox_inches='tight')
  plt.show()

  print("===== Feature extraction details =====")
  print("Features extracted: Acceleration Slopes, FFT Equal Band Powers, Barometer Slope.")

def plot_confusion_matrix(confusion_matrix, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """ This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusion_matrix)

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.0f'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":

  data_path = './uploaded/'
  # Change it to your net id
  my_netid = 'cuu25pbu'

  raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + my_netid)


  "***Plot Raw Data***"
  # Plot the raw data to get a sense about what features might work.
  # You can comment out this line of code if you don't want to see the plots
  plot_raw_data(raw_data_features, raw_data_labels)


  "***Feature Extraction***"
  features, labels = feature_extraction(raw_data_features, raw_data_labels, timestamps)


  "***Plot Features***"
  # You can comment out this line of code if you don't want to see the plots
  plot_extracted_features(features, labels)


  "***Classify User's Own Data***"
  five_fold_cross_validation(features, labels)


  "***Person-independent model (i.e. train on other's data and test on your own data)***"
  X_train = None
  Y_train = None
  X_test = None
  Y_test = None

  dirs = os.listdir(data_path)
  for dir in dirs:
    print("Processing data from %s" %dir)
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
    evaluate_generalized_model(X_train, Y_train, X_test, Y_test)
  else:
    print(
      "No training data from other subjects found. Skipping person-independent model evaluation. Please add other users' data to the 'uploaded' directory to train the model.")