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

    print("\n--- Running Method 1 Pipeline ---")
    print("Extracting features and training model...")
    features1, labels1 = fe1(raw_data_features, raw_data_labels, timestamps)
    clf1 = DecisionTreeClassifier()
    clf1.fit(features1, labels1.ravel())

    print("Benchmarking FULL DATASET feature extraction + inference memory...")
    tracemalloc.start()
    f1_full, _ = fe1(raw_data_features, raw_data_labels, timestamps)
    clf1.predict(f1_full)
    current1, peak1 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    current_mb_1 = current1 / (1024 * 1024)
    peak_mb_1 = peak1 / (1024 * 1024)

    print("\n--- Running Method 2 Pipeline ---")
    print("Extracting features and training model...")
    features2, labels2 = fe2(raw_data_features, raw_data_labels, timestamps)
    clf2 = DecisionTreeClassifier()
    clf2.fit(features2, labels2.ravel())

    print("Benchmarking FULL DATASET feature extraction + inference memory...")
    tracemalloc.start()
    f2_full, _ = fe2(raw_data_features, raw_data_labels, timestamps)
    clf2.predict(f2_full)
    current2, peak2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    current_mb_2 = current2 / (1024 * 1024)
    peak_mb_2 = peak2 / (1024 * 1024)

    print("\n" + "="*50)
    print("                  MEMORY COMPARISON")
    print("="*50)
    print(f"Method 1 (first_process_character) features shape:  {features1.shape}")
    print(f"Method 2 (second_processing_character) features shape: {features2.shape}")
    print("-"*50)
    print(f"Method 1 Net Memory Allocation:  {current_mb_1:.6f} MB")
    print(f"Method 1 Peak Memory Allocation: {peak_mb_1:.6f} MB")
    print(f"Method 2 Net Memory Allocation:  {current_mb_2:.6f} MB")
    print(f"Method 2 Peak Memory Allocation: {peak_mb_2:.6f} MB")
    print("="*50)

    print("\n--- Peak Memory Comparison ---")
    if peak_mb_1 > peak_mb_2:
        ratio = peak_mb_1 / peak_mb_2 if peak_mb_2 > 0 else float('inf')
        print(f"Method 2 uses {ratio:.2f}x less PEAK memory than Method 1 on full dataset.")
    elif peak_mb_2 > peak_mb_1:
        ratio = peak_mb_2 / peak_mb_1 if peak_mb_1 > 0 else float('inf')
        print(f"Method 1 uses {ratio:.2f}x less PEAK memory than Method 2 on full dataset.")
    else:
        print("Both methods use the same amount of peak memory.")

    print("\n--- Net (Total Process) Memory Comparison ---")
    if current_mb_1 > current_mb_2:
        ratio = current_mb_1 / current_mb_2 if current_mb_2 > 0 else float('inf')
        print(f"Method 2 uses {ratio:.2f}x less NET memory than Method 1 on full dataset.")
    elif current_mb_2 > current_mb_1:
        ratio = current_mb_2 / current_mb_1 if current_mb_1 > 0 else float('inf')
        print(f"Method 1 uses {ratio:.2f}x less NET memory than Method 2 on full dataset.")
    else:
        print("Both methods use the same amount of net memory.")

    # Visualization
    import numpy as np
    labels = ['Method 1\n(Full Dataset)', 'Method 2\n(Full Dataset)']
    current_usage = [current_mb_1, current_mb_2]
    peak_usage = [peak_mb_1, peak_mb_2]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, current_usage, width, label='Net Memory', color='skyblue')
    rects2 = ax.bar(x + width/2, peak_usage, width, label='Peak Memory', color='lightcoral')

    ax.set_ylabel('Memory Allocation (MB)')
    ax.set_title('Full Dataset Net and Peak Memory Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.6f} MB',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    os.makedirs('./process_time', exist_ok=True)
    save_path = './process_time/processing_memory_table.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    compare_training_memory()
