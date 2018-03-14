import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
import time

from sklearn.ensemble import RandomForestClassifier
x = pickle.load(open("Saved_Data/Train_Examples.pickle", "rb"))
y_true = pickle.load(open("Saved_Data/Train_Labels.pickle", "rb"))

start = time.time()
hyperparams = { 'n_estimators':[20,25,30,35,40]}
#hyperparams = {'learning_rate': np.logspace(0.0001,1,8), 'n_estimators': np.linspace(50,200,8), 'subsample':[0.6,0.7,0.8,0.9,1.0],'max_depth':[2,3,4,5]}
#hyperparams = {'learning_rate':[0.001,0.01], 'n_estimators': [100,150], 'subsample':[0.8]}

regressor = RandomForestClassifier()

cross_val_obj = GridSearchCV(regressor,hyperparams,verbose=1,refit=True)
cross_val_obj.fit(x,y_true)

end = time.time()

with open("Saved_Data/RandomForestCVResults.pickle",'wb') as f:
    pickle.dump(cross_val_obj,f)


print(cross_val_obj.best_score_)
print(cross_val_obj.best_params_)

print(end - start)

x_test = pickle.load(open("Saved_Data/Test_Examples.pickle", "rb"))
y_test=  pickle.load(open("Saved_Data/Test_Labels.pickle", "rb"))

cross_val_obj.score(x_test,y_test)


print("Finish")

