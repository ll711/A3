import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

def feature_extraction(raw_data_features, raw_data_labels, timestamps):
  """    raw_data_features: The fourth column is the barometer data.

  Returns:
    features: Features extracted from the data features, where
              features[:, 0:3] is the slope of 3-axis acceleration (x, y, z);
              features[:, 3:7] is the fft power spectrum of equally-spaced frequencies;
              features[:, 7] is the slope of pressure.
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
  plt.figure(figsize=(12, 15))

  # Plot the acceleration X axis slope
  plt.subplot(6, 1, 1)
  plt.plot(features[:, 0], label='X-Slope', color='blue')
  plt.legend(loc='upper right', fontsize=8)
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.ylabel('Slope', fontsize=8)
  plt.gca().set_title('Slope of Accelerometer X-axis', fontsize=8)

  # Plot the acceleration Y axis slope
  plt.subplot(6, 1, 2)
  plt.plot(features[:, 1], label='Y-Slope', color='cyan')
  plt.legend(loc='upper right', fontsize=8)
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.ylabel('Slope', fontsize=8)
  plt.gca().set_title('Slope of Accelerometer Y-axis', fontsize=8)

  # Plot the acceleration Z axis slope
  plt.subplot(6, 1, 3)
  plt.plot(features[:, 2], label='Z-Slope', color='magenta')
  plt.legend(loc='upper right', fontsize=8)
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.ylabel('Slope', fontsize=8)
  plt.gca().set_title('Slope of Accelerometer Z-axis', fontsize=8)

  # Plot the barometer slope
  plt.subplot(6, 1, 4)
  plt.plot(features[:, -1], color='orange')
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.ylabel('mbar/s', fontsize=8)
  plt.gca().set_title('Rate of Change of Barometric Pressure (Slope)', fontsize=8)

  # Plot one of the FFT Equal Band Powers
  plt.subplot(6, 1, 5)
  plt.plot(features[:, 3], color='green')
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.ylabel('FFT Power', fontsize=8)
  plt.gca().set_title('FFT Equispaced Band Power (First Band)', fontsize=8)

  plt.subplot(6, 1, 6)
  plt.plot(labels, color='red')
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.ylabel('Activity', fontsize=8)
  plt.gca().set_title('Activity Label', fontsize=8)
  plt.grid(True)
  plt.tight_layout()

  os.makedirs('./data_processing', exist_ok=True)
  plt.savefig('./data_processing/extracted_features.png', bbox_inches='tight')
  plt.show()


def plot_feature_importances(features, labels):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels.ravel())
    importances = clf.feature_importances_

    feature_names = ['X-Slope', 'Y-Slope', 'Z-Slope'] + [f'FFT Band {i}' for i in range(features.shape[1]-4)] + ['Barometer Slope']

    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importances (Random Forest)')
    plt.tight_layout()

    os.makedirs('./data_processing', exist_ok=True)
    plt.savefig('./data_processing/feature_importances.png', bbox_inches='tight')
    plt.show()

