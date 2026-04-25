import os
import time
from sklearn.tree import DecisionTreeClassifier
from activity_recognition2 import compute_raw_data
from first_process_character import feature_extraction as fe1
from second_processing_character import feature_extraction as fe2
import matplotlib.pyplot as plt

def compare_training_time():
    data_path = './uploaded/'
    my_netid = 'cuu25pbu'

    print("Loading raw data...")
    try:
        raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + my_netid)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the 'uploaded' directory contains the expected data.")
        return

    print("\n--- Running Method 1 Pipeline ---")
    print("Extracting features and training model...")
    features1, labels1 = fe1(raw_data_features, raw_data_labels, timestamps)
    clf1 = DecisionTreeClassifier()
    clf1.fit(features1, labels1.ravel())

    print("Benchmarking FULL DATASET feature extraction + inference...")
    start_time_1 = time.time()
    f1_full, _ = fe1(raw_data_features, raw_data_labels, timestamps)
    clf1.predict(f1_full)
    end_time_1 = time.time()
    avg_time_1 = end_time_1 - start_time_1

    print("\n--- Running Method 2 Pipeline ---")
    print("Extracting features and training model...")
    features2, labels2 = fe2(raw_data_features, raw_data_labels, timestamps)
    clf2 = DecisionTreeClassifier()
    clf2.fit(features2, labels2.ravel())

    print("Benchmarking FULL DATASET feature extraction + inference...")
    start_time_2 = time.time()
    f2_full, _ = fe2(raw_data_features, raw_data_labels, timestamps)
    clf2.predict(f2_full)
    end_time_2 = time.time()
    avg_time_2 = end_time_2 - start_time_2

    print("\n" + "="*50)
    print("                   TIME COMPARISON")
    print("="*50)
    print(f"Method 1 full features shape:  {features1.shape}")
    print(f"Method 2 full features shape: {features2.shape}")
    print("-"*50)
    print(f"Method 1 Full Dataset (Extract + Predict) Time: {avg_time_1:.6f} seconds")
    print(f"Method 2 Full Dataset (Extract + Predict) Time: {avg_time_2:.6f} seconds")
    print("="*50)

    if avg_time_1 > avg_time_2:
        print(f"Method 2 is {(avg_time_1/avg_time_2):.2f}x faster than Method 1 on full dataset.")
    elif avg_time_2 > avg_time_1:
        print(f"Method 1 is {(avg_time_2/avg_time_1):.2f}x faster than Method 2 on full dataset.")

    # Visualization
    methods = ['Method 1\n(Full Dataset)', 'Method 2\n(Full Dataset)']
    times = [avg_time_1, avg_time_2]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, times, color=['skyblue', 'lightgreen'])
    plt.ylabel('Total Time (Seconds)')
    plt.title('Full Dataset Total Time (Extract + Predict)')

    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.6f} s',
                 ha='center', va='bottom', fontsize=10)

    os.makedirs('./process_time', exist_ok=True)
    save_path = './process_time/processing_times_table.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    compare_training_time()
