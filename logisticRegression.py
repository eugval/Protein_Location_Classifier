import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from utils import load_data

from globals import data_folder, models_folder





def LRCV(models_folder,data_folder,save_suffix,extension,cv=3):
    x,y_true,x_test,y_test = load_data(data_folder, save_suffix, extension)


    hyperparams = [{'C': [0.1,0.5,1.0], 'penalty': ['l1'], 'solver':['saga'],'class_weight':['balanced',None], 'max_iter' : [300]},
                     {'C': [0.1,0.5,1.0], 'penalty': ['l2'], 'solver':['newton-cg','sag'],'class_weight':['balanced',None], 'max_iter' :[300]} ]
    #hyperparams = {'learning_rate': np.logspace(0.0001,1,8), 'n_estimators': np.linspace(50,200,8), 'subsample':[0.6,0.7,0.8,0.9,1.0],'max_depth':[2,3,4,5]}
    #hyperparams = {'learning_rate':[0.001,0.01], 'n_estimators': [100,150], 'subsample':[0.8]}

    regressor = LogisticRegression( )

    cross_val_obj = GridSearchCV(regressor,hyperparams,verbose=2,refit=True, cv=cv,n_jobs=-1)
    cross_val_obj.fit(x,y_true)



    with open("{}LRCV{}{}".format(models_folder,save_suffix,extension),'wb') as f:
        pickle.dump(cross_val_obj,f)

    cv_score = cross_val_obj.best_score_
    best_params = cross_val_obj.best_params_
    test_score = cross_val_obj.score(x_test, y_test)

    print(test_score)
    print(cv_score)
    print(best_params)
    print("Finish")

    return test_score, cv_score, best_params


if(__name__=="__main__"):
    save_suffix = "_mid_complexity"
    extension = ".pickle"
    LRCV(models_folder,data_folder,save_suffix,extension)