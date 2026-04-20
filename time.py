import time
import os
import numpy as np
import tracemalloc

# Override plt.show to avoid blocking execution during timing
import matplotlib.pyplot as plt
plt.show = lambda: None

import activity_recognition2 as ar2
import activity_recognition3 as ar3

def run_ar2_workflow():
    data_path = './uploaded/'
    my_netid = 'cuu25pbu'

    raw_data_features, raw_data_labels, timestamps = ar2.compute_raw_data(data_path + my_netid)
    features, labels = ar2.feature_extraction(raw_data_features, raw_data_labels, timestamps)

    start_time = time.time()
    ar2.five_fold_cross_validation(features, labels)
    cv_time = time.time() - start_time

    X_train = None
    Y_train = None
    X_test = None
    Y_test = None

    if os.path.exists(data_path):
        dirs = os.listdir(data_path)
        for dir_name in dirs:
            dir_path = os.path.join(data_path, dir_name)
            if not os.path.isdir(dir_path):
                continue
            raw_data_feat, raw_labels, ts = ar2.compute_raw_data(dir_path)
            feat, lab = ar2.feature_extraction(raw_data_feat, raw_labels, ts)

            if dir_name == my_netid:
                X_test = feat
                Y_test = lab
            else:
                if X_train is None:
                    X_train = feat
                    Y_train = lab
                else:
                    X_train = np.append(X_train, feat, axis=0)
                    Y_train = np.append(Y_train, lab, axis=0)

    gen_time = 0
    if X_train is not None and Y_train is not None:
        start_time = time.time()
        ar2.evaluate_generalized_model(X_train, Y_train, X_test, Y_test)
        gen_time = time.time() - start_time

    return cv_time, gen_time

def measure_time_ar2():
    print("Pass 1: Measuring time for activity_recognition2 (without tracemalloc overhead)...")
    cv_time, gen_time = run_ar2_workflow()

    print(f"[AR2] Cross Validation Time: {cv_time:.4f} seconds")
    print(f"[AR2] Generalized Model Time: {gen_time:.4f} seconds")
    print(f"[AR2] Total Model Time: {cv_time + gen_time:.4f} seconds\n")
    return cv_time, gen_time

def measure_memory_ar2():
    print("Pass 2: Measuring memory for activity_recognition2...")
    tracemalloc.start()
    run_ar2_workflow()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024

    print(f"[AR2] Peak Memory Usage: {peak_mb:.4f} MB\n")
    return peak_mb

def run_ar3_workflow():
    data_path = './uploaded/'
    my_netid = 'cuu25pbu'

    raw_data_features, raw_data_labels, timestamps = ar3.compute_raw_data(data_path + my_netid)
    features, labels = ar3.feature_extraction(raw_data_features, raw_data_labels, timestamps)

    start_time = time.time()
    ar3.five_fold_cross_validation(features, labels)
    cv_time = time.time() - start_time

    X_train = None
    Y_train = None
    X_test = None
    Y_test = None

    if os.path.exists(data_path):
        dirs = os.listdir(data_path)
        for dir_name in dirs:
            dir_path = os.path.join(data_path, dir_name)
            if not os.path.isdir(dir_path):
                continue
            raw_data_feat, raw_labels, ts = ar3.compute_raw_data(dir_path)
            feat, lab = ar3.feature_extraction(raw_data_feat, raw_labels, ts)

            if dir_name == my_netid:
                X_test = feat
                Y_test = lab
            else:
                if X_train is None:
                    X_train = feat
                    Y_train = lab
                else:
                    X_train = np.append(X_train, feat, axis=0)
                    Y_train = np.append(Y_train, lab, axis=0)

    gen_time = 0
    if X_train is not None and Y_train is not None:
        start_time = time.time()
        ar3.evaluate_generalized_model(X_train, Y_train, X_test, Y_test)
        gen_time = time.time() - start_time

    return cv_time, gen_time

def measure_time_ar3():
    print("Pass 1: Measuring time for activity_recognition3 (without tracemalloc overhead)...")
    cv_time, gen_time = run_ar3_workflow()

    print(f"[AR3] Cross Validation Time: {cv_time:.4f} seconds")
    print(f"[AR3] Generalized Model Time: {gen_time:.4f} seconds")
    print(f"[AR3] Total Model Time: {cv_time + gen_time:.4f} seconds\n")
    return cv_time, gen_time

def measure_memory_ar3():
    print("Pass 2: Measuring memory for activity_recognition3...")
    tracemalloc.start()
    run_ar3_workflow()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024

    print(f"[AR3] Peak Memory Usage: {peak_mb:.4f} MB\n")
    return peak_mb

if __name__ == '__main__':
    os.makedirs('process_time', exist_ok=True)

    # First calculate and plot Time
    cv2, gen2 = measure_time_ar2()
    cv3, gen3 = measure_time_ar3()

    time_table_data = [
        ["Model", "Cross Validation Time (s)", "Generalized Model Time (s)", "Total Time (s)"],
        ["Activity Recognition 2", f"{cv2:.4f}", f"{gen2:.4f}", f"{cv2 + gen2:.4f}"],
        ["Activity Recognition 3", f"{cv3:.4f}", f"{gen3:.4f}", f"{cv3 + gen3:.4f}"]
    ]

    fig1, ax1 = plt.subplots(figsize=(8, 2))
    ax1.axis('tight')
    ax1.axis('off')

    table1 = ax1.table(cellText=time_table_data, loc='center', cellLoc='center', colColours=["#f0f0f0"]*4)
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.5)

    for (i, j), cell in table1.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d9d9d9')

    plt.savefig('process_time/processing_times_table.png', bbox_inches='tight', dpi=300)
    print("Visualized time table saved to process_time/processing_times_table.png\n")

    # Then calculate and plot Memory
    mem2 = measure_memory_ar2()
    mem3 = measure_memory_ar3()

    mem_table_data = [
        ["Model", "Peak Memory (MB)"],
        ["Activity Recognition 2", f"{mem2:.4f}"],
        ["Activity Recognition 3", f"{mem3:.4f}"]
    ]

    fig2, ax2 = plt.subplots(figsize=(6, 2))
    ax2.axis('tight')
    ax2.axis('off')

    table2 = ax2.table(cellText=mem_table_data, loc='center', cellLoc='center', colColours=["#f0f0f0"]*2)
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.5)

    for (i, j), cell in table2.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d9d9d9')

    plt.savefig('process_time/processing_memory_table.png', bbox_inches='tight', dpi=300)
    print("Visualized memory table saved to process_time/processing_memory_table.png")
