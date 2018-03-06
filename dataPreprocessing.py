##########
#Indices:
# cyto : 0
# mito : 1
# nuc : 2
# secr : 3
############
from Bio import SeqIO
import pickle

from featureMaps import dictionary_feature_extractor, global_and_sliding_window_feature_dict,global_feature_dict

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split




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


# encodes feature dictionaries as numpy vectors, needed by scikit-learn.
vectorizer = DictVectorizer(sparse=True)


x = vectorizer.fit_transform([dictionary_feature_extractor(rec) for rec in x_raw])
x_new = vectorizer.fit_transform([global_feature_dict(rec,False) for rec in x_raw])


print((x==x_new))

X_train, X_test, y_train, y_test = train_test_split( x, y_true, test_size=0.1)

with open("Saved_Data/Train_Examples.pickle",'wb') as f:
    pickle.dump(X_train,f)


with open("Saved_Data/Train_Labels.pickle",'wb') as f:
    pickle.dump(y_train,f)

with open("Saved_Data/Test_Examples.pickle",'wb') as f:
    pickle.dump(X_test,f)

with open("Saved_Data/Test_Labels.pickle",'wb') as f:
    pickle.dump(y_test,f)









