import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
import time

from sklearn.svm import SVC

x = pickle.load(open("Saved_Data/Train_Examples.pickle", "rb"))
y_true = pickle.load(open("Saved_Data/Train_Labels.pickle", "rb"))

start = time.time()
hyperparams = {'C': [0.75,0.8,0.85], 'kernel':['rbf'],"probability":[True]}



regressor = SVC()

cross_val_obj = GridSearchCV(regressor,hyperparams,verbose=2,refit=True)
cross_val_obj.fit(x,y_true)

end = time.time()

with open("Saved_Data/SVMCVResults.pickle",'wb') as f:
    pickle.dump(cross_val_obj,f)


print(cross_val_obj.best_score_)
print(cross_val_obj.best_params_)

print(end - start)

x_test = pickle.load(open("Saved_Data/Test_Examples.pickle", "rb"))
y_test =  pickle.load(open("Saved_Data/Test_Labels.pickle", "rb"))

cross_val_obj.score(x_test,y_test)


print("Finish")
