import pickle


x_test = pickle.load(open("Saved_Data/Test_Examples.pickle", "rb"))
y_test=  pickle.load(open("Saved_Data/Test_Labels.pickle", "rb"))
cross_val_obj = pickle.load(open("Saved_Data/GradientBoostingCVResults.pickle",'rb'))

estimator = cross_val_obj.best_estimator_

y_pred = estimator.predict_proba(x_test)

print(y_pred)