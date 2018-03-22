import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from utils import load_data

from globals import data_folder, models_folder


def SVMCV(models_folder,data_folder,save_suffix,extension, cv = 3):
    x,y_true,x_test,y_test = load_data(data_folder, save_suffix, extension)

    hyperparams = {'C': [0.75,0.8,0.85], 'kernel':['rbf'],"probability":[True]}

    regressor = SVC()

    cross_val_obj = GridSearchCV(regressor,hyperparams,verbose=2,refit=True , cv=cv,n_jobs=-1)
    cross_val_obj.fit(x,y_true)



    with open("{}SVMCV{}{}".format(models_folder,save_suffix,extension),'wb') as f:
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
    save_suffix = "_tunned_features"
    extension = ".pickle"
    SVMCV(models_folder, data_folder, save_suffix, extension)

