from dataPreprocessing import dataPreprocessing, data_split
import pickle
from globals import data_folder, models_folder
from gradientBoosting import GBCV
from SVM import SVMCV
from randomForest import RFCV
from logisticRegression import LRCV
from ensembler import ensembleClassifier
from tuneFeatures import tune_features

save_suffix= "_tunned_features"
extension = ".pickle"
test_size = 0.1

x_raw_train, x_raw_test, y_train, y_test  = data_split(test_size)

print("Tunning features....")
#feat_params = tune_features(x_raw_train,y_train, extension)
feat_params = pickle.load(open("Saved_Data/Features/Best_features.pickle",'rb'))

print(feat_params)

print("Making Features....")
for key in feat_params.keys():
    feat_params[key]=feat_params[key][0]

x_train, x_test, y_train, y_test = dataPreprocessing(x_raw_train, x_raw_test,y_train,y_test,save_suffix,data_folder,extension, **feat_params)

print("Cross validating...")

gb_test_score, _, _ = GBCV(models_folder,data_folder,save_suffix,extension, cv=5)

print("GBCV finished")
print(gb_test_score)

sv_test_score, _, _ = SVMCV(models_folder,data_folder,save_suffix,extension, cv=5)

print("SVMCV finished")
print(sv_test_score)

rf_test_score,_,_ = RFCV(models_folder,data_folder,save_suffix,extension,cv=5)

print("Random forest finished")
print(rf_test_score)

lr_test_score, _,_ = LRCV(models_folder,data_folder,save_suffix,extension,cv=5)


test_score, y_pred,y_pred_proba = ensembleClassifier(models_folder,data_folder,save_suffix,extension)

print("Ensembler finished")
print(test_score)



