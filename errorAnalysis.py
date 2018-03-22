import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import label_binarize
from globals import CLASSES
from errorAnalysisAPI import make_characterisation_histograms , plot_roc_curves, plot_confusion_matrix,plot_feature_importances
from sklearn.model_selection import StratifiedKFold
from globals import data_folder,models_folder, error_folder
from utils import load_data
from ensembler import ensembleClassifier
import numpy as np
from time import time

save_suffix= "_tunned_features"
regressor_str = "Ensembler"
extension = ".pickle"

CV = False
Curves = False
Hists = False
FeatureRankGB = False
FeatureRankRF = False


x_train, y_train, x_test, y_test = load_data(data_folder,save_suffix,extension)

if(CV):
    cross_validator = StratifiedKFold(n_splits=10)

    arr_accuracy = []
    arr_f1_cyto = []
    arr_f1_mito = []
    arr_f1_nuc = []
    arr_f1_secr = []
    arr_f1_macro = []
    arr_f1_micro = []
    arr_precision_macro =[]
    arr_recall_macro = []

    for index, cv_index in cross_validator.split(x_train, y_train):
        print("new cross validation loop")
        start = time()
        x = x_train[index]
        y = np.array(y_train)[index]

        x_cv = x_train[cv_index]
        y_cv = np.array(y_train)[cv_index]

        acc, y_pred, _ = ensembleClassifier(models_folder,data_folder,save_suffix,extension,x=x,y_true=y,x_test=x_cv,y_test=y_cv, save=False)

        arr_accuracy.append(acc)

        f1_cyto, f1_mito,f1_nuc,f1_secr = f1_score(y_cv,y_pred, average =None)
        f1_macro = f1_score(y_cv,y_pred,average='macro')
        f1_micro = f1_score(y_cv,y_pred,average='micro')

        arr_f1_cyto.append(f1_cyto)
        arr_f1_mito.append(f1_mito)
        arr_f1_nuc.append(f1_nuc)
        arr_f1_secr.append(f1_secr)
        arr_f1_micro.append(f1_micro)
        arr_f1_macro.append(f1_macro)

        precision_macro = precision_score(y_cv,y_pred,average='macro')
        recall_macro = recall_score(y_cv,y_pred,average='macro')

        arr_precision_macro.append(precision_macro)
        arr_recall_macro.append(recall_macro)
        end = time()
        print("Time ellapsed : {}".format(end-start))





    cv_accuracy = np.mean(arr_accuracy)
    cv_std_accuracy = np.std(arr_accuracy)

    print("accuracy :{} std: {}".format(cv_accuracy,cv_std_accuracy))
    print(arr_accuracy)

    cv_f1_macro = np.mean(arr_f1_macro)
    cv_std_f1_macro= np.std(arr_f1_macro)
    print("f1_macro :{} std: {}".format(cv_f1_macro,cv_std_f1_macro))
    print(arr_f1_macro)
    #cv_f1_micro = np.mean(arr_f1_micro)
    #cv_std_f1_micro= np.std(arr_f1_micro)
    #print("f1_micro :{} std: {}".format(cv_f1_micro,cv_std_f1_micro))
    #print(arr_f1_micro)
    cv_prec_macro = np.mean(arr_precision_macro)
    cv_std_prec_macro= np.std(arr_precision_macro)
    print("precision macro :{} std: {}".format(cv_prec_macro,cv_std_prec_macro))
    print(arr_precision_macro)
    cv_rec_macro = np.mean(arr_recall_macro)
    cv_std_rec_macro= np.std(arr_recall_macro)
    print("recall macro :{} std: {}".format(cv_rec_macro,cv_std_rec_macro))
    print(arr_recall_macro)


    cv_f1_cyto = np.mean(arr_f1_cyto)
    cv_std_f1_cyto = np.std(arr_f1_cyto)
    print("f1_cyto :{} std: {}".format(cv_f1_cyto,cv_std_f1_cyto))
    print(arr_f1_cyto)
    cv_f1_mito = np.mean(arr_f1_mito)
    cv_std_f1_mito = np.std(arr_f1_mito)
    print("f1_mito :{} std: {}".format(cv_f1_mito,cv_std_f1_mito))
    print(arr_f1_mito)
    cv_f1_nuc = np.mean(arr_f1_nuc)
    cv_std_f1_nuc = np.std(arr_f1_nuc)
    print("f1_nuc :{} std: {}".format(cv_f1_nuc,cv_std_f1_nuc))
    print(arr_f1_nuc)
    cv_f1_secr = np.mean(arr_f1_secr)
    cv_std_f1_secr = np.std(arr_f1_secr)
    print("f1_secr :{} std: {}".format(cv_f1_secr,cv_std_f1_secr))
    print(arr_f1_secr )




classifier = pickle.load(open(models_folder+regressor_str+save_suffix+extension,'rb'))
y_pred = classifier.predict(x_test)
test_acc = classifier.score(x_test,y_test)
print(test_acc)

print(classification_report(y_test,y_pred,digits = 3, target_names=CLASSES))

if(Curves):
    confusion_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(confusion_matrix,CLASSES,True)
    plt.savefig(error_folder+regressor_str + "_confusion_matrix"+save_suffix+".png")

    y_test_binary = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_proba = classifier.predict_proba(x_test)

    plt.figure()

    plot_roc_curves(y_proba,y_test_binary)
    plt.savefig(error_folder+regressor_str + "_ROC_curves"+save_suffix+".png")



x_test = pickle.load(open("{}Test_Examples{}.pickle".format(data_folder,save_suffix), "rb"))

vectorizer = pickle.load(open("{}Vectorizer{}.pickle".format(data_folder,save_suffix), "rb"))
x_test_dicts = vectorizer.inverse_transform(x_test)

disp_feat = 10
if(FeatureRankGB):

    Gradient_Boost_CV = pickle.load(open("{}{}{}{}".format(models_folder,"GBCV",save_suffix,extension),'rb'))
    GB = Gradient_Boost_CV.best_estimator_
    title = "Overall Feature importances"
    plot_feature_importances(GB,vectorizer,disp_feat,title )

    for proclass in range(4):
        title = "Feature Importances for {} proteins".format(CLASSES[proclass])
        plot_feature_importances(GB, vectorizer, disp_feat, title)


if(FeatureRankRF):
    Forest_CV = pickle.load(open("{}{}{}{}".format(models_folder,"RFCV",save_suffix,extension),'rb'))

    RF = Forest_CV.best_estimator_
    plot_feature_importances(RF,vectorizer,disp_feat, RF= True)


#make_characterisation_histograms(x_test_dicts,"sequence_length",y_pred,y_test,"sequence length")
make_characterisation_histograms(x_test_dicts,"hydrophobicity_start",y_pred,y_test, "average hydrophobicity at the start of the sequence")
make_characterisation_histograms(x_test_dicts,"helix_global",y_pred,y_test, "overall proportion of aminoacids prone to form an alpha-helix")
make_characterisation_histograms(x_test_dicts,"sheet_global",y_pred,y_test, "overall proportion of aminoacids prone to form a beta-sheet")
make_characterisation_histograms(x_test_dicts,"isoelectric_point_global",y_pred,y_test, "average isoelectric point")
#make_characterisation_histograms(x_test_dicts,"molecular_weight_global",y_pred,y_test, "molecular weight")
#make_characterisation_histograms(x_test_dicts,"side_chain_charge_global",y_pred,y_test, "average side chain charge")
plt.show()

print("finish")

