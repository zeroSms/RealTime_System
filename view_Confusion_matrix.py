import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=['others', 'nod', 'shake'], columns=['others', 'nod', 'shake'])

    sns.heatmap(df_cmx, square=True, cbar=True, annot=True, cmap='Blues', fmt='d')
    # plt.figure(figsize=(10, 7))
    plt.yticks(rotation=0)
    plt.xlabel('Predict Label', fontsize=12, rotation=0)
    plt.ylabel('True Label', fontsize=12)
    plt.show()