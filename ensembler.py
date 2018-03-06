import pickle

x = pickle.load(open("Saved_Data/Train_Examples.pickle", "rb"))
y_true = pickle.load(open("Saved_Data/Train_Labels.pickle", "rb"))

x_test = pickle.load(open("Saved_Data/Test_Examples.pickle", "rb"))
y_test =  pickle.load(open("Saved_Data/Test_Labels.pickle", "rb"))



Gradient_Boost_CV = pickle.load(open("Saved_Data/GradientBoostingCVResults.pickle",'rb'))
Logisic_CV = pickle.load(open("Saved_Data/LogisticRegressionCVResults.pickle",'rb'))
Forest_CV = pickle.load(open("Saved_Data/RandomForestCVResults.pickle",'rb'))
SVMCV = pickle.load(open("Saved_Data/SVMCVResults.pickle",'rb'))

gb_param =
lr_param =

estimator = cross_val_obj.best_estimator_

y_pred = estimator.predict_proba(x_test)

print(y_pred)