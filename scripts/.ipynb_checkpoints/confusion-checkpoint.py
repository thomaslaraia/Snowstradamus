from scripts.FSC_dataframe_phoreal import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def plot_confusion_matrix(true_labels, predicted_labels, classes, save=None, plot=True):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    if plot == False:
        return accuracy

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix\nAccuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(accuracy, precision, recall, f1))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save != None:
        plt.savefig(f'./images{save}')
    plt.show()
    return accuracy

def confusion(loc_df, FI, variable, decision = 'asr', save = None, plot=True, preset=True):

    if variable == 'FSC':
        classes = ['No Snow', 'Snow']
    else:
        classes = ['No Snow', 'Ground Snow', 'Ground and Canopy Snow']

    truth = loc_df[variable].values

    def assign_value(row):
        for i, value in enumerate(FI[:-1]):
            if row[decision] < value:
                return i
        return i+1

    prediction = loc_df.apply(lambda row: assign_value(row), axis=1).values

    acc = plot_confusion_matrix(truth, prediction, classes, save, plot)
    return acc