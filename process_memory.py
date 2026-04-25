import os
import tracemalloc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from activity_recognition2 import compute_raw_data
from first_process_character import feature_extraction as fe1
from second_processing_character import feature_extraction as fe2

def compare_training_memory():
    data_path = './uploaded/'
    my_netid = 'cuu25pbu'

    print("Loading raw data...")
    try:
        raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + my_netid)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the 'uploaded' directory contains the expected data.")
        return

    print("Extracting features for Method 1...")
    features1, labels1 = fe1(raw_data_features, raw_data_labels, timestamps)

    print("Extracting features for Method 2...")
    features2, labels2 = fe2(raw_data_features, raw_data_labels, timestamps)

    iterations = 50

    print(f"\nBenchmarking model training and prediction memory over {iterations} iterations...")

    # Benchmark Method 1
    clf1 = DecisionTreeClassifier()
    tracemalloc.start()
    for _ in range(iterations):
        clf1.fit(features1, labels1.ravel())
        clf1.predict(features1)
    _, peak1 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb_1 = peak1 / (1024 * 1024)

    # Benchmark Method 2
    clf2 = DecisionTreeClassifier()
    tracemalloc.start()
    for _ in range(iterations):
        clf2.fit(features2, labels2.ravel())
        clf2.predict(features2)
    _, peak2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb_2 = peak2 / (1024 * 1024)

    print("\n" + "="*50)
    print("                  MEMORY COMPARISON")
    print("="*50)
    print(f"Method 1 (first_process_character) features shape:  {features1.shape}")
    print(f"Method 2 (second_processing_character) features shape: {features2.shape}")
    print("-"*50)
    print(f"Method 1 Peak Memory Allocation: {peak_mb_1:.6f} MB")
    print(f"Method 2 Peak Memory Allocation: {peak_mb_2:.6f} MB")
    print("="*50)

    if peak_mb_1 > peak_mb_2:
        ratio = peak_mb_1 / peak_mb_2 if peak_mb_2 > 0 else float('inf')
        print(f"Method 2 uses {ratio:.2f}x less memory than Method 1.")
    elif peak_mb_2 > peak_mb_1:
        ratio = peak_mb_2 / peak_mb_1 if peak_mb_1 > 0 else float('inf')
        print(f"Method 1 uses {ratio:.2f}x less memory than Method 2.")
    else:
        print("Both methods use the same amount of memory.")

    # Visualization
    methods = ['Method 1', 'Method 2']
    memory_usage = [peak_mb_1, peak_mb_2]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, memory_usage, color=['lightcoral', 'mediumpurple'])
    plt.ylabel('Peak Memory Allocation (MB)')
    plt.title('Model Training and Prediction Peak Memory Comparison')

    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.6f} MB',
                 ha='center', va='bottom', fontsize=10)

    os.makedirs('./process_time', exist_ok=True)
    save_path = './process_time/processing_memory_table.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    compare_training_memory()

