
'''
This function is taken from Scikit-learns implementation from :
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
'''



import numpy as np
from scipy import interp
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from globals import CLASSES
import matplotlib.pyplot as plt

def plot_roc_curves(y_proba, y_test_binary):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    #fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binary.ravel(), np.array(y_proba).ravel())
   # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])




    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(CLASSES))]))
    lw = 2
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(CLASSES)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(CLASSES)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves

    #plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.2f})'
      #             ''.format(roc_auc["micro"]),
     #        color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(len(CLASSES)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(CLASSES[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for each of the classes and micro and macro averages.')
    plt.legend(loc="lower right")








'''
This function is taken from Scikit-learns implementation from :
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
'''
import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def make_characterisation_histograms(x_test_dicts, feature,y_pred,y_test, title_feature, ratio = True):
    missclassifed_examples_hist = []
    examples_hist = []
    for index, values in enumerate(zip(y_pred,y_test)):
        prediction,label = values
        if(prediction != label):
            missclassifed_examples_hist.append(x_test_dicts[index][feature])
        examples_hist.append(x_test_dicts[index][feature])

    if(ratio):
        hist_all,bins = np.histogram(examples_hist, bins = 10)
        hist_missclassified, bins= np.histogram(missclassifed_examples_hist, range = (min(examples_hist), max(examples_hist)), bins = bins)

        hist_all[hist_all==0]=1
        final_hist = np.divide(hist_missclassified.astype(float), hist_all)

        plt.figure()
        plt.bar(bins[:-1], final_hist, width=(max(bins)-min(bins))/len(bins))
        plt.xlim(min(bins), max(bins))
        plt.title("Histogram of the proportion of missclassified proteins per {}".format(title_feature))
        plt.ylabel("Proportion of missclassified proteins")
        plt.xlabel("{} values".format(title_feature))

    else:
        plt.figure()
        plt.hist(missclassifed_examples_hist)
        plt.title("Histogram of the number of missclassified proteins as a function of {} ".format(title_feature))
        plt.ylabel("Number of missclassified proteins")
        plt.xlabel("{} values".format(title_feature))

        plt.figure()
        plt.hist(examples_hist)
        plt.title("Histogram of the number of proteins as a function of {}p".format(title_feature))
        plt.ylabel("Number of  proteins")
        plt.xlabel("{} values".format(title_feature))



def plot_feature_importances(classifier, vectorizer,disp_feat, title,proclass = False, RF = False):
    if(not proclass):
        importances = classifier.feature_importances_
    else:
        importances = np.mean([tree.feature_importances_ for tree in classifier.estimators_[:,proclass]],
                                       axis=0)


    '''
    if(not RF):
        stds = np.zeros((4, importances.size))
        for class_label in range(4):
                stds[class_label] = np.std([tree.feature_importances_ for tree in classifier.estimators_[:,class_label]],
                                           axis=0)
        if(not proclass):
            std = np.mean(stds, axis=0)
        else:
            std = stds[proclass]

    else:
        std= np.std([tree.feature_importances_ for tree in classifier.estimators_],
                               axis=0)
    '''



    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(disp_feat):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title(title)
    plt.bar(range(disp_feat), importances[indices[:disp_feat]],
            color="b",  align="center") #yerr=std[indices[:disp_feat]],
    plt.xticks(range(disp_feat), [vectorizer.feature_names_[index] for index in indices[:disp_feat]], rotation=-45)
    plt.xlim([-1, disp_feat])
    plt.show()






