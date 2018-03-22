from Bio import SeqIO
import pickle
from featureMaps import  global_and_sliding_window_feature_dict,global_feature_dict
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


from globals import data_folder

save_suffix = "_mid_complexity"
extension = ".pickle"
test_size = 0.1

if(save_suffix=="_basic"):
    feat_params = { "global_bipeptide":False}
elif(save_suffix=="_mid_complexity"):
    feat_params = {"global_bipeptide": True}


def data_split(test_size):
    cyto = list(SeqIO.parse("./Data/cyto.fasta", "fasta"))
    mito = list(SeqIO.parse("./Data/mito.fasta", "fasta"))
    nuc = list(SeqIO.parse("./Data/nucleus.fasta", "fasta"))
    secr = list(SeqIO.parse("./Data/secreted.fasta", "fasta"))

    cyto_labels = [0]*len(cyto)
    mito_labels = [1]*len(mito)
    nuc_labels = [2]*len(nuc)
    secr_labels = [3]*len(secr)


    x_raw = cyto+mito+nuc+secr
    y_true = cyto_labels+mito_labels+nuc_labels+secr_labels


    x_raw_train, x_raw_test, y_train, y_test= train_test_split(x_raw, y_true, test_size=test_size)
    return x_raw_train, x_raw_test,y_train,y_test


def dataPreprocessing(x_raw_train, x_raw_test,y_train, y_test,save_suffix,data_folder,extension, save=True, **kwargs):

    # encodes feature dictionaries as numpy vectors, needed by scikit-learn.
    vectorizer = DictVectorizer(sparse=True)


    if(save_suffix == "_window"):
        x_train = vectorizer.fit_transform([global_and_sliding_window_feature_dict(rec,100,False) for rec in x_raw_train])
        x_test = vectorizer.fit_transform([global_and_sliding_window_feature_dict(rec,100,False) for rec in x_raw_test])
    else:
        x_train = vectorizer.fit_transform([global_feature_dict(rec,**kwargs) for rec in x_raw_train])
        x_test = vectorizer.fit_transform([global_feature_dict(rec, **kwargs) for rec in x_raw_test])



    if(save):
        with open("{}Train_Examples{}{}".format(data_folder,save_suffix,extension),'wb') as f:
            pickle.dump(x_train,f)

        with open("{}Train_Labels{}{}".format(data_folder,save_suffix,extension),'wb') as f:
            pickle.dump(y_train,f)

        with open("{}Test_Examples{}{}".format(data_folder,save_suffix,extension),'wb') as f:
            pickle.dump(x_test,f)

        with open("{}Test_Labels{}{}".format(data_folder,save_suffix,extension),'wb') as f:
            pickle.dump(y_test,f)

        with open("{}Vectorizer{}{}".format(data_folder,save_suffix,extension),'wb') as f:
            pickle.dump(vectorizer,f)

    return x_train, x_test , y_train, y_test



if(__name__=="__main__"):


    x_raw_train, x_raw_test, y_train, y_test =  data_split(test_size)
    x_train, x_test, y_train,y_test = dataPreprocessing(x_raw_train, x_raw_test,y_train,y_test,save_suffix,data_folder,extension, **feat_params)

    print("done")




