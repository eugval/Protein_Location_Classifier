import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier ,RandomForestClassifier
from sklearn.svm import SVC
from utils import  load_data
from sklearn.model_selection import GridSearchCV


from globals import data_folder,models_folder

save_suffix ="_mid_complexity"
extension = ".pickle"

do_CV = False

def ensembleClassifier(models_folder,data_folder,save_suffix,extension,x=None,y_true=None,x_test=None,y_test=None, save=True ):
    if(x==None):
      x,y_true,x_test,y_test = load_data(data_folder, save_suffix, extension)

    print("loading models")
    Gradient_Boost_CV = pickle.load(open("{}{}{}{}".format(models_folder,"GBCV",save_suffix,extension),'rb'))
    Logistic_CV = pickle.load(open("{}{}{}{}".format(models_folder,"LRCV",save_suffix,extension),'rb'))
    Forest_CV = pickle.load(open("{}{}{}{}".format(models_folder,"RFCV",save_suffix,extension),'rb'))
    SVMCV = pickle.load(open("{}{}{}{}".format(models_folder,"SVMCV",save_suffix,extension),'rb'))

    gb_param = Gradient_Boost_CV.best_params_
    lr_param = Logistic_CV.best_params_
    rf_param = Forest_CV.best_params_
    sv_param = SVMCV.best_params_

    gb = GradientBoostingClassifier(**gb_param)
    rf = RandomForestClassifier(**rf_param)
    lr = LogisticRegression(**lr_param)
    sv = SVC(**sv_param)

    classifier = VotingClassifier(estimators=[('gb',gb),('lr',lr),('rf',rf), ('sv',sv)],voting="soft")


    print("start fitting my model...")
    classifier.fit(x,y_true)

    if(save):
        with open("{}Ensembler{}{}".format(models_folder,save_suffix,extension),'wb') as f:
            pickle.dump(classifier,f)

    "finishing off..."
    test_score = classifier.score(x_test,y_test)
    y_pred = classifier.predict(x_test)
    y_pred_proba = classifier.predict_proba(x_test)

    print(test_score)

    return test_score, y_pred,y_pred_proba



def ensembleClassifierCV(models_folder,data_folder,save_suffix,extension):
    x, y_true, x_test, y_test = load_data(data_folder, save_suffix, extension)

    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier()
    lr = LogisticRegression()
    sv = SVC()

    classifier = VotingClassifier(estimators=[('gb', gb), ('lr', lr), ('rf', rf), ('sv', sv)], voting="soft")

    hyperparams = {'gb__learning_rate': [0.0001,0.001], 'gb__n_estimators':[100,150,200], 'gb__subsample':[0.8,1.0],'gb__max_depth':[3,4],
                   'lr__C': [0.5, 0.8], 'lr__penalty': ['l2'], 'lr__solver': ['newton-cg', 'sag'], 'lr__class_weight': ['balanced', None], 'lr__max_iter': [300],
                   'rf__n_estimators':[50,100,150],
                   'sv__C': [0.75, 0.8, 0.85], 'sv__kernel': ['rbf'], "sv__probability": [True]}


    cross_val_obj = GridSearchCV(classifier,hyperparams, verbose=2,refit=True)
    cross_val_obj.fit(x,y_true)

    with open("{}ensembleClassifierCV{}{}".format(models_folder,save_suffix,extension),'wb') as f:
        pickle.dump(cross_val_obj,f)

    cv_score = cross_val_obj.best_score_
    best_params = cross_val_obj.best_params_
    test_score = cross_val_obj.score(x_test, y_test)
    y_pred = cross_val_obj.predict(x_test)
    y_pred_proba = cross_val_obj.predict_proba(x_test)

    print(test_score)
    print(cv_score)
    print(best_params)
    print("Finish")

    return y_pred, y_pred_proba, test_score, cv_score, best_params



if( __name__=="__main__"):
    if(do_CV):
        ensembleClassifierCV(models_folder, data_folder, save_suffix, extension)
    else:
        ensembleClassifier(models_folder,data_folder,save_suffix,extension)
