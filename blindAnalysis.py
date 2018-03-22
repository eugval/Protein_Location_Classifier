from Bio import SeqIO
import pickle
from sklearn.feature_extraction import DictVectorizer
from globals import data_folder, models_folder,error_folder
from featureMaps import  global_and_sliding_window_feature_dict,global_feature_dict
from globals import CLASSES_S

save_suffix= "_tunned_features"
regressor_str = "Ensembler"
extension = ".pickle"


blind_x = list(SeqIO.parse("./Data/blind.fasta", "fasta"))

blind_x_ids = [record.id for record in blind_x]

vectorizer = DictVectorizer(sparse=True)

feat_params = pickle.load(open("Saved_Data/Features/Best_features.pickle",'rb'))

print("Making Features....")
for key in feat_params.keys():
    feat_params[key]=feat_params[key][0]

x_test = vectorizer.fit_transform([global_feature_dict(rec,**feat_params) for rec in blind_x])

classifier = pickle.load(open(models_folder+regressor_str+save_suffix+extension,'rb'))
y_pred = classifier.predict(x_test)
y_pred_proba = classifier.predict_proba(x_test)


final_predictions = []

for i in range(len(y_pred)):
    final_predictions.append((blind_x_ids[i],CLASSES_S[y_pred[i]],y_pred_proba[i][y_pred[i]]))

with open("{}Final_Predictions{}{}".format(error_folder,save_suffix,extension),'wb') as f:
    pickle.dump(final_predictions,f)


for element in final_predictions:
   # print("{} {} Confidence {} \ \ ".format(element[0],element[1],element[2]))
    print("textbf{} & {} & {} \ \ ".format(element[0],element[1],element[2]))