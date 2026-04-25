import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score
from activity_recognition2 import compute_raw_data
from first_process_character import feature_extraction as fe1
from second_processing_character import feature_extraction as fe2
from train_model import plot_confusion_matrix

def evaluate_model(features, labels, activity_indices, title_prefix="Method"):
    true_labels = []
    predicted_labels = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Convert labels to 1D early for stratified splitting in new versions of sklearn
    labels = labels.ravel()

    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y=y_train)
        y_pred = clf.predict(X_test)

        predicted_labels.extend(y_pred)
        true_labels.extend(y_test)

    # Generate and plot Confusion Matrix
    confusion_matrix_obj = np.zeros((len(activity_indices), len(activity_indices)))
    for i in range(len(true_labels)):
        confusion_matrix_obj[int(true_labels[i]), int(predicted_labels[i])] += 1

    print(f"\n===== Confusion Matrix ({title_prefix}) =====")
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(
        confusion_matrix_obj,
        list(activity_indices.keys()),
        normalize=False,
        title=f"Confusion Matrix ({title_prefix})"
    )

    os.makedirs('./process_time', exist_ok=True)
    cm_save_path = f'./process_time/confusion_matrix_{title_prefix.replace(" ", "_")}.png'
    plt.savefig(cm_save_path, bbox_inches='tight')
    print(f"Confusion matrix for {title_prefix} saved to {cm_save_path}")
    plt.show()

    acc = accuracy_score(true_labels, predicted_labels)
    rec = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    return acc, rec

def compare_accuracy_recall():
    data_path = './uploaded/'
    my_netid = 'cuu25pbu'

    activity_indices = {
        'Stationary': 0,
        'Walking-flat-surface': 1,
        'Walking-up-stairs': 2,
        'Walking-down-stairs': 3,
        'Elevator-up': 4,
        'Running': 5,
        'Elevator-down': 6
    }

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

    print("\nEvaluating models using 5-Fold Cross Validation...")

    acc1, rec1 = evaluate_model(features1, labels1, activity_indices, title_prefix="Method 1")
    acc2, rec2 = evaluate_model(features2, labels2, activity_indices, title_prefix="Method 2")

    print("\n" + "="*60)
    print("                ACCURACY & RECALL COMPARISON")
    print("="*60)
    print(f"Method 1 (first_process) Accuracy: {acc1:.4f}  | Recall (Macro): {rec1:.4f}")
    print(f"Method 2 (second_process) Accuracy: {acc2:.4f}  | Recall (Macro): {rec2:.4f}")
    print("="*60)

    # Comparison Print
    if acc1 > acc2:
        print(f"Method 1 has HIGHER Accuracy (+{acc1 - acc2:.4f})")
    elif acc2 > acc1:
        print(f"Method 2 has HIGHER Accuracy (+{acc2 - acc1:.4f})")
    else:
        print("Both methods have the SAME Accuracy")

    if rec1 > rec2:
        print(f"Method 1 has HIGHER Recall (+{rec1 - rec2:.4f})")
    elif rec2 > rec1:
        print(f"Method 2 has HIGHER Recall (+{rec2 - rec1:.4f})")
    else:
        print("Both methods have the SAME Recall")

    # Visualization
    labels = ['Accuracy', 'Recall (Macro)']
    m1_scores = [acc1, rec1]
    m2_scores = [acc2, rec2]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, m1_scores, width, label='Method 1', color='skyblue')
    rects2 = ax.bar(x + width/2, m2_scores, width, label='Method 2', color='lightgreen')

    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison (Accuracy & Recall)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)  # scores are 0.0 ~ 1.0
    ax.legend(loc='lower right')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    os.makedirs('./process_time', exist_ok=True)
    save_path = './process_time/processing_accurate_table.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    compare_accuracy_recall()
