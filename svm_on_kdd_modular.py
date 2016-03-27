import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as time1

from sklearn import datasets
from sklearn.svm import SVC
from sklearn import cross_validation
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

class MySVM(object):
    def __init__(self, num_features=None):
            if num_features is None:
                    num_features = []
            else:
                    self.num_features = num_features
            #print('asd')
            
    def preprocess_data(self,kdd_data_10percent):        
            ##--- PK1: First Lets classify the each packets into either normal or attack, thus convert all labels other than 'normal' to 'attack'
            #num_features = self.num_features
            features = kdd_data_10percent[self.num_features].astype(float)
            labels = kdd_data_10percent['label'].copy()
            labels[labels!='normal.'] = 1 # all types of attacks are labled as 1
            labels[labels=='normal.'] = 0
            #print(labels.value_counts())

            kdd_data_10percent['label'] = labels
            #kdd_data_10percent.to_csv('beforescale.csv')
            ## -- PK2: Scale the data
            from sklearn import preprocessing
            min_max_scalar = preprocessing.MinMaxScaler()
            X_train_minmax = min_max_scalar.fit_transform(features)
            #X_Scaled = preprocessing.scale(features)
            #np.setprintoption(precision=2)
            #X_train_minmax = np.asarray(X_train_minmax)
            
            df = pd.DataFrame(X_train_minmax)
            kdd_data_10percent[self.num_features] = df
            reduced_matrix = self.apply_PCA_on(kdd_data_10percent)
            kdd_data_10percent.to_csv('afterscale.csv')
            #print('X_Scaled:\t',X_Scaled)
            #from sklearn.preprocessing import MinMaxScaler
            #features.apply(lambda x:MinMaxScaler(feature_range=(0,1)).fit_transform(x))
            #print('Scaled Features: ',features)
            
            #kdd_data_10percent[self.num_features] = features
            #print(kdd_data_10percent.describe())
            return reduced_matrix
    def apply_PCA_on(self,kdd_data_10percent):
        training_input_data = np.asarray(kdd_data_10percent[num_features]) # load this from KDD data
        myData = np.array(training_input_data)
        from matplotlib.mlab import PCA
        results = PCA(myData,standardize=False)
        return results.Y
    def writeToFile(self,filename,df):
        #print(df)
        df.to_csv(filename)
            
    def loadDataSet(self,filename):
            print('Processing the File: ',filename)
            col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
                                     "logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                                     "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                                     "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                                     "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                                     "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
            #sample of 5000 records: N_records_file.data
            #kddcup.data_10_percent_corrected
            kdd_data_10percent = pd.read_csv(filename,header=None,names=col_names)
            return kdd_data_10percent
       
    def plot_confusion_matrix(self,cm,title='Confusion Matrix',cmap=plt.cm.Blues):
            plt.imshow(cm,interpolation='nearest',cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks,['normal','attack'],rotation=45)
            plt.yticks(tick_marks,['normal','attack'])
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')    

    def split_training_test_data(self,kdd_data_10percent,split):    
       # -- PK3: Training classifier uses SVM classifier --
            from sklearn.cross_validation import train_test_split
            features_train, features_test, labels_train, labels_test = train_test_split(
                    kdd_data_10percent[self.num_features],kdd_data_10percent['label'],test_size=split,random_state=42)
            return features_train, features_test, labels_train, labels_test
            
    def train(self,features_train,labels_train):
            clf = SVC(kernel='rbf',C=1)
            print("Started Training the System",time1.ctime())
            train_t0 = time1.time()
            clf.fit(features_train,labels_train)
            train_t1 = time1.time()
            print("End Training the System",time1.ctime())
            print("Trainig Phase: {} records are trained in {} ".format(len(features_train),train_t1-train_t0))
            return clf
    def train_cross_validationset(self,features_train,labels_train):
            clf = SVC(kernel='rbf',C=1)
            print("Started Training the System",time1.ctime())
            train_t0 = time1.time()
            from sklearn.cross_validation import cross_val_score
            scores = cross_validation.cross_val_score(clf,features_train,labels_train,cv=5)
            print('Scores : \t',scores)
            train_t1 = time1.time()
            print("End Training the System",time1.ctime())
            print("Trainig Phase: {} records are trained in {} ".format(len(features_train),train_t1-train_t0))
            print("Accuray: %0.4f (+/- %0.4f)"%(scores.mean(),scores.std()*2))
        

    def test(self,clf,features_test):
            test_t0 = time1.time()
            print("Started Testing the System : ",time1.ctime())
            prediction = clf.predict(features_test)
            test_t1 = time1.time()
            print("End Testing the System :",time1.ctime())
            print("Testing Phase: {} records are tested in {} ".format(len(features_test),test_t1-test_t0))
            return prediction
    #def apply_PCA_on_KDD(self,kdd_DS):
        

    def calculate_accuracy(self,prediction,labels_test):
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(prediction,labels_test)
        return acc

    def produce_classification_report(self,y_true,y_pred):
        print('Y_true {} Y-pred {}'.format(repr(y_true),repr(y_pred)))
        from sklearn.metrics import classification_report
        target_names = [0,1]
        print('\n************************** Classification Report ****************************')
        print(classification_report(repr(y_true),repr(y_pred)))

    def confusion_matrix_pandas(self,y_true,y_pred):
        y_tr_pd = pd.Series(y_true,name='actual')
        y_pred_pd = pd.Series(y_pred,name='predicted')
        df_confusion = pd.crosstab(y_tr_pd,y_pred_pd,rownames=['Actual'],colnames=['Predicted'],margins=True)
        return df_confusion
        #print(df_confusion)
        

    def main(self):
            filename="200000_records_file.data"#"kddcup.data_10_percent_corrected"#"20000_records_file.data"
            split=0.4
            
            kdd_data_10percent = self.loadDataSet(filename)
            reduced_matrix = self.preprocess_data(kdd_data_10percent) # this incldes considering numerical(continous data) and scaling it in the range of 0-1
            print('##########MATRIX SHAPE is {}',np.asarray(reduced_matrix).shape)
            kdd_data_10percent[self.num_features] = reduced_matrix
            #kdd_data_10percent.to_csv('afterPCA_afterscale.csv')
            features_train, features_test, labels_train, labels_test = self.split_training_test_data(kdd_data_10percent,split)
            #print(np.asarray(labels_test))
            clf = self.train(features_train,labels_train)
            prediction = self.test(clf,features_test)
            y_true = np.asarray(labels_test)
            y_pred = prediction
            #self.produce_classification_report(y_true,y_pred)
            print('\n************************** Confusion Matrix ****************************')
            from sklearn.metrics import confusion_matrix
            #cm = confusion_matrix(y_true,y_pred)
            cm = self.confusion_matrix_pandas(y_true,y_pred)
            print(cm)
            #np.set_printoptions(precision=2)
            #plt.figure()
            #self.plot_confusion_matrix(cm)
            #plt.show()
            print('\n************************** END **********************************')
            #acc = self.calculate_accuracy(prediction,np.asarray(labels_test))
            #print("R Squared is {}.".format(round(acc,5)))

            # Or Using Cross Validation Set
            #self.train_cross_validationset(features_train,labels_train)
            #compute confusion matrix
            '''cm = confusion_matrix(labels_test,prediction)
            np.set_printoptions(precision=2)
            print(cm)
            plt.figure()
            self.plot_confusion_matrix(cm)
            plt.show()'''


num_features = ["duration","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
                     "logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                     "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                     "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                     "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                     "dst_host_rerror_rate","dst_host_srv_rerror_rate"] 

mySVM = MySVM(num_features[:])
mySVM.main()
