import numpy as np
from collections import Counter

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

        segment_fft_powers = np.abs(np.fft.fft(segment, axis=0)) ** 2

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

