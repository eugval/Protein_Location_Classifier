import pickle

def load_data(folder,save_suffix,extension):

    x = pickle.load(open("{}Train_Examples{}{}".format(folder, save_suffix, extension), "rb"))
    y_true = pickle.load(open("{}Train_Labels{}{}".format(folder, save_suffix, extension), "rb"))

    x_test = pickle.load(open("{}Test_Examples{}{}".format(folder, save_suffix, extension), "rb"))
    y_test = pickle.load(open("{}Test_Labels{}{}".format(folder, save_suffix, extension), "rb"))

    return x, y_true, x_test, y_test