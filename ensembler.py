import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier ,RandomForestClassifier
from sklearn.svm import SVC

save_suffix ="_mid_complexity"

x = pickle.load(open("Saved_Data/Train_Examples{}.pickle".format(save_suffix), "rb"))
y_true = pickle.load(open("Saved_Data/Train_Labels{}.pickle".format(save_suffix), "rb"))

x_test = pickle.load(open("Saved_Data/Test_Examples{}.pickle".format(save_suffix), "rb"))
y_test =  pickle.load(open("Saved_Data/Test_Labels{}.pickle".format(save_suffix), "rb"))



Gradient_Boost_CV = pickle.load(open("Saved_Data/GradientBoostingCVResults.pickle",'rb'))
Logistic_CV = pickle.load(open("Saved_Data/LogisticRegressionCVResults.pickle",'rb'))
Forest_CV = pickle.load(open("Saved_Data/RandomForcestCVResults.pickle",'rb'))
SVMCV = pickle.load(open("Saved_Data/SVMCVResults.pickle",'rb'))

gb_param = Gradient_Boost_CV.best_params_
lr_param = Logistic_CV.best_params_
rf_param = Forest_CV.best_params_
sv_param = SVMCV.best_params_

gb = GradientBoostingClassifier(**gb_param)
rf = RandomForestClassifier(**rf_param)
lr = LogisticRegression(**lr_param)
sv = SVC(**sv_param)

regressor = VotingClassifier(estimators=[('gb',gb),('lr',lr),('rf',rf),('sv',sv)],voting="soft")

regressor.fit(x,y_true)

score = regressor.score(x_test,y_test)
print(score)

y_pred = regressor.predict_proba(x_test)


with open("Saved_Data/Ensembler{}.pickle".format(save_suffix),'wb') as f:
    pickle.dump(regressor,f)