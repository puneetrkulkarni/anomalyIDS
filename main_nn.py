import network
import numpy as np
import time
import pandas as pd
import ConfigParams


# Data Loader starts
def loadDataSet(filename):
        #sample of 5000 records: N_records_file.data
        #kddcup.data_10_percent_corrected
        kdd_data_10percent = pd.read_csv(filename,header=None,names=ConfigParams.col_names)
        return kdd_data_10percent

def preprocess_data(kdd_data_10percent):        
        ##--- PK1: First Lets classify the each packets into either normal or attack, thus convert all labels other than 'normal' to 'attack'
        #num_features = self.num_features
        
        features = kdd_data_10percent[ConfigParams.numerical_features].astype(float)
        labels = kdd_data_10percent['label'].copy()
        #labels[labels!='normal.'] = 'attack.'
        
        labels[labels!='normal.'] = 1
        labels[labels=='normal.'] = 0
        kdd_data_10percent['label'] = labels.copy()
        #labels['normal'].replace(0)
        #labels['attack'].replace(1)
            
        print(labels.value_counts())
        #mapping = {'normal': 1,'attack':-1}
        #kdd_data_10percent['label'] = labels
        #kdd_data_10percent.replace({'normal':mapping,'attack':mapping})
        #print(kdd_data_10percent['labels'].value_counts())
       # kdd_data_10percent.replace({'normal':mapping,'attack':mapping})
        ## -- PK2: Scale the data
        from sklearn.preprocessing import MinMaxScaler
        features=features.apply(lambda x:MinMaxScaler().fit_transform(x))
        kdd_data_10percent[ConfigParams.numerical_features] = features.copy()
        #print(kdd_data_10percent[ConfigParams.numerical_features].values)

def get_sturctured_data(kdd_data_10percent):
   
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train,labels_test = train_test_split(kdd_data_10percent[ConfigParams.numerical_features],kdd_data_10percent['label'],test_size=0.33,random_state=42)
    x_train_array = np.asarray(features_train)
    y_train_array = np.asarray(labels_train)
    #print('x_train_array shape:',x_train_array.shape)
    x_train_array1 = [np.reshape(x,(38,1)) for x in x_train_array]
    y_train_array1 = [np.reshape(x,(1,1)) for x in y_train_array]
   
    print('x_train_array1 shape:',np.asarray(x_train_array1).shape)
    print('y_train_array1 shape:',np.asarray(y_train_array1).shape)
    x_test_array = np.asarray(features_test)
    y_test_array = np.asarray(labels_test)
    
    x_test_array1 = [np.reshape(x,(38,1)) for x in x_test_array]
    y_test_array1 = [np.reshape(x,(1,1)) for x in y_test_array]
    
    print('x_test_array1 shape:',np.asarray(x_test_array1).shape)
    print('y_test_array1 shape:',np.asarray(y_test_array1).shape)
        
    
    training_data = zip(x_train_array1,y_train_array)
    testing_data = zip(x_test_array1,y_test_array)
    return (training_data,testing_data)


# Let's load the data and instatiate the network eg 5000_records_file.data
kdd_data_10percent = loadDataSet(ConfigParams.filename)
preprocess_data(kdd_data_10percent)
training_data,testing_data = get_sturctured_data(kdd_data_10percent)
print('No of layers:',ConfigParams.no_of_layers)
net = network.Network([ConfigParams.no_of_inputs,ConfigParams.processing_units_hiddenlayer_1,ConfigParams.no_of_outputs])
print('Dataset File Name:',ConfigParams.filename)
print('No. of Epoches for Training:',ConfigParams.epoches)
print('Update Batch Size:',ConfigParams.batch_size)
print('Learning Rate(eta):',ConfigParams.eta)

print('Dataset File Name:',ConfigParams.filename)
net.SGD(training_data,ConfigParams.epoches,ConfigParams.batch_size,ConfigParams.eta,testing_data)

print("Done with NN training phase")
