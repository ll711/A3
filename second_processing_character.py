import os
import numpy as np
from collections import Counter

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
