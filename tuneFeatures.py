from Bio import SeqIO
import pickle
from featureMaps import global_feature_dict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier ,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from dataPreprocessing import data_split


from globals import data_folder



def tune_features(x_raw_train, x_raw_test, y_train, y_test):


    features_dict ={"amount_start":[5,10,15,20,30,40,50,60,70],"amount_in_end":[5,10,15,20,30,40,50,60,70], "global_bipeptide":[True,False],"local_bipeptide":[False,True],"aromaticity":[True,False],"instability":[True,False], "average_h":[True,False],"side_charge_ave": [True,False], "gravy": [True,False]}

    #features_dict = {"amount_start":[5]}


    best_features_dict ={"amount_start":[0,0],"amount_in_end":[0,0], "global_bipeptide":[0,0],"local_bipeptide":[0,0],"aromaticity":[0,0],"instability":[0,0],"average_h":[0,0],"side_charge_ave": [0,0], "gravy": [0,0]}


    x_raw_train, x_raw_val, y_train, y_val = train_test_split(x_raw_train, y_train, test_size=0.1)



    for feature in features_dict.keys():
        for value in features_dict[feature]:
            print("Trying {} = {}".format(feature,value))


            # encodes feature dictionaries as numpy vectors, needed by scikit-learn.
            vectorizer = DictVectorizer(sparse=True)

            use = {feature:value}

            x_train = vectorizer.fit_transform([global_feature_dict(rec,**use) for rec in x_raw_train])
            x_test = vectorizer.fit_transform([global_feature_dict(rec,**use) for rec in x_raw_test])


            print("Training classifier...")

            gb = GradientBoostingClassifier()
            rf = RandomForestClassifier()
            lr = LogisticRegression()
            sv = SVC(probability=True)

            classifier = VotingClassifier(estimators=[('gb', gb), ('lr', lr), ('rf', rf), ('sv', sv)], voting="soft")

            classifier.fit(x_train, y_train)

            score = classifier.score(x_test,y_test)

            if(score>best_features_dict[feature][1]):
                best_features_dict[feature][1]=score
                best_features_dict[feature][0]=value



    print(best_features_dict)

    with open("{}Best_features{}".format(data_folder, extension), 'wb') as f:
        pickle.dump(best_features_dict, f)




    print("finished")
    return best_features_dict


if(__name__=="__main__"):
    extension = ".pickle"
    test_size = 0.1
    x_raw_train, x_raw_test, y_train, y_test = data_split(test_size)
    tune_features(x_raw_train, x_raw_test, y_train, y_test)



