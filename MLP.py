from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import Sequential

from keras.layers import Dense
import pickle







import collections
import matplotlib.pyplot as plt

class TrainingTracker(object):
    '''
    Track training performance metrics
    '''
    def __init__(self):
        self.valDict = collections.defaultdict(lambda: [])

    def addArray(self, valArray, dataLabel = None):
        '''
        Add a full array of values in the tracker object under a metric label
        :param valArray: The array of values to add
        :param dataLabel: What are the values added
        :return: N/A
        '''

        if(dataLabel == None):
            raise ValueError("How do you want me to track data if you don't provide labels for them?")
        elif(not isinstance(dataLabel,str)):
            raise ValueError("Seriously, not a string?! ...")
        else:
            self.valDict[dataLabel] = valArray

    def makePlots(self, save=False):
        '''
        Plot the performance metrics.
        :return: N/A
        '''

        plt.figure(figsize=(20,20))
        number_of_plots = len(self.valDict.keys())

        i=1
        for k,v in self.valDict.items():
            plt.subplot(number_of_plots, 1, i)
            plt.plot(v)
            plt.xlabel("Epochs")
            plt.ylabel(k)
            plt.title(k +" vs Epoch Number")
            i+=1

        plt.tight_layout()
        if(save):
            plt.savefig("../SavedData/" + save)
        else:
            plt.show()



    def getValues(self, label):
        '''Get the values associated with the label'''
        return self.valDict[label]

    def getFullDict(self):
        '''Get the full tracker dict'''
        return self.valDict











x = pickle.load(open("Saved_Data/Train_Examples.pickle", "rb"))
y_true = pickle.load(open("Saved_Data/Train_Labels.pickle", "rb"))

x_test = pickle.load(open("Saved_Data/Test_Examples.pickle", "rb"))
y_test = pickle.load(open("Saved_Data/Test_Labels.pickle", "rb"))


y_true = np.array(y_true)
y_train = np.zeros((y_true.size, y_true.max()+1))
y_train[np.arange(y_true.size),y_true] = 1

y_test = np.array(y_test)
y_test_label= np.zeros((y_test.size, y_test.max()+1))
y_test_label[np.arange(y_test.size),y_test] = 1


x=x.todense()

model = Sequential()
model.add(Dense(units=32, activation='elu', kernel_initializer='glorot_normal', input_dim=106))

model.add(Dense(units=4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

history = model.fit(x, y_train, epochs=5, batch_size=128)

returns = model.predict(x)

loss_and_metrics = model.evaluate(x_test, y_test_label, batch_size=128)

tracker = TrainingTracker()
tracker.addArray(history.history['acc'], 'Training_Accuracy')
tracker.addArray(history.history['loss'], 'Training_Cost')
tracker.makePlots()
print(loss_and_metrics)