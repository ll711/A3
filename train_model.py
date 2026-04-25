import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE

def print_metrics_from_confusion_matrix(conf_matrix, class_names, save_path=None, title="Metrics Table"):
    """Print and optionally save per-class precision, recall, F1 as an image with a title."""
    n_classes = conf_matrix.shape[0]
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = tp.sum() / conf_matrix.sum()

    print(f"\n===== {title} =====")
    print(f"{'Activity':<25} {'Precision':>10} {'Recall':>10} {'F1-score':>10}")
    for i, name in enumerate(class_names):
        print(f"{name:<25} {precision[i]:10.4f} {recall[i]:10.4f} {f1[i]:10.4f}")
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("=============================================\n")

    if save_path is not None:
        cell_text = []
        for i, name in enumerate(class_names):
            cell_text.append([f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}"])
        cell_text.append([f"{accuracy:.4f}", "", ""])

        rows = class_names + ['Overall Accuracy']
        columns = ['Precision', 'Recall', 'F1-score']

        fig, ax = plt.subplots(figsize=(6, len(rows) * 0.5 + 1.5))
        ax.axis('off')
        ax.set_title(title, fontsize=12, weight='bold', pad=10)
        table = ax.table(cellText=cell_text,
                         rowLabels=rows,
                         colLabels=columns,
                         cellLoc='center',
                         rowLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Metrics table saved to {save_path}")

def plot_confusion_matrix(confusion_matrix, classes, normalize=False,
                          title='Confusion Matrix', cmap=plt.cm.Blues):
    """Prints and plots the confusion matrix."""
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

def five_fold_cross_validation(features, labels, activity_indices):
    """
    Within-subject evaluation using 5-fold cross-validation.
    Only the user's own data is used (no additional participants' data).
    """
    true_labels = []
    predicted_labels = []

    for train_index, test_index in StratifiedKFold(n_splits=5).split(features, labels):
        X_train = features[train_index, :]
        Y_train = labels[train_index].ravel()
        X_test = features[test_index, :]
        Y_test = labels[test_index].ravel()

        clf = DecisionTreeClassifier()
        clf.fit(X_train, Y_train)
        predicted_label = clf.predict(X_test)

        predicted_labels += predicted_label.flatten().tolist()
        true_labels += Y_test.flatten().tolist()

    confusion_matrix = np.zeros((len(activity_indices), len(activity_indices)))
    for i in range(len(true_labels)):
        confusion_matrix[int(true_labels[i]), int(predicted_labels[i])] += 1

    print("===== Confusion Matrix (Within-subject, no additional data) =====")
    plot_confusion_matrix(confusion_matrix, list(activity_indices.keys()), normalize=False)
    os.makedirs('./data_process', exist_ok=True)
    plt.savefig('./data_process/confusion_matrix_within_subject.png', bbox_inches='tight')
    plt.show()

    print("\nWithin-subject model evaluation (To Do 4):")
    print_metrics_from_confusion_matrix(confusion_matrix,
                                        list(activity_indices.keys()),
                                        save_path='./data_process/metrics_within_no_extra.png',
                                        title='Within-subject Model (No Additional Data)')

def evaluate_generalized_model(X_train, Y_train, X_test, Y_test, activity_indices, plot_feature_boxplots):
    """
    Between-subjects evaluation: train on other participants' data,
    test on the target user's data. Uses additional data from other subjects.
    """
    clf = DecisionTreeClassifier().fit(X_train, y=Y_train.ravel())
    Y_pred = clf.predict(X_test)

    confusion_matrix = np.zeros((len(activity_indices), len(activity_indices)))
    for i in range(len(Y_test)):
        confusion_matrix[int(Y_test[i]), int(Y_pred[i])] += 1

    print("===== Confusion Matrix (Between-subjects, with additional data) =====")
    plot_confusion_matrix(confusion_matrix, list(activity_indices.keys()))
    os.makedirs('./data_process', exist_ok=True)
    plt.savefig('./data_process/confusion_matrix_between_subjects.png', bbox_inches='tight')
    plt.show()

    print("\nBetween-subjects model evaluation (To Do 5):")
    print_metrics_from_confusion_matrix(confusion_matrix,
                                        list(activity_indices.keys()),
                                        save_path='./data_process/metrics_between_with_extra.png',
                                        title='Between-subjects Model (With Additional Data)')

    feature_names = [
        'Accel Mean', 'Accel Variance',
        'FFT Equal Band 1', 'FFT Equal Band 2', 'FFT Equal Band 3', 'FFT Equal Band 4',
        'FFT Log Band 1', 'FFT Log Band 2', 'FFT Log Band 3', 'FFT Log Band 4',
        'FFT Log Band 5', 'FFT Log Band 6', 'FFT Log Band 7',
        'Barometer Slope'
    ]

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances from Decision Tree")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.savefig('./data_process/feature_importances.png', bbox_inches='tight')
    plt.show()

    selector = RFE(estimator=clf, n_features_to_select=5)
    selector.fit(X_train, Y_train.ravel())
    print("===== Mask of Top-5 Features =====")
    for i, support in enumerate(selector.support_):
        if support:
            print(f"- {feature_names[i]}")

    print("===== Generating Feature Boxplots on Combined General Training Data =====")
    plot_feature_boxplots(X_train, Y_train, filename='training_feature_boxplots.png')
