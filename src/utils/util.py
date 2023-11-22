import matplotlib.pyplot as plt
import seaborn as sns


def plot_matrix(matrix, classes, x_label, y_label, save_to, ticks_rotation=45, show=False):
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(matrix, annot=True, cmap='crest', fmt='d')
    classes_indexes = classes.argsort()
    classes_labels = classes.tolist()
    ax.set_xticks(classes_indexes + 0.5)
    ax.set_yticks(classes_indexes + 0.5)
    ax.set_xticklabels(classes_labels, rotation=ticks_rotation, ha='left', rotation_mode='anchor')
    ax.set_yticklabels(classes_labels, rotation=ticks_rotation, ha='left', rotation_mode='anchor')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_to, bbox_inches='tight', pad_inches=0.5, dpi=600)
    if show:
        plt.show()


def plot_roc(fpr, tpr, save_to, ticks_rotation=45, show=False):
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, label='AUC = %.2f')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(save_to, bbox_inches='tight', pad_inches=0.5, dpi=600)
    if show:
        plt.show()
