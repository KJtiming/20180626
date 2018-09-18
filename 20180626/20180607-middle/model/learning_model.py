from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import config
import logging




path = './data'

#train_data_size = 500
train_data_size = 400
#nb_epoch = 500
nb_epoch = 10
is_multioutput = False  # (single-output target:serving Rx) / (multi-output target:serving Rx+neighbor Rx)
begin_time = end_time = 0
pci_num = 4

itri_data = path + '/51-5F/Data300.csv'

#NN
def build_dNN_model(X_train, y_train, nb_feature = 2, pci_only = True):
    print "+++Deep NN"
    enb_feature_num = config.GERNAL['enb_feature_num']
    feature = int(enb_feature_num) + int(nb_feature)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim = feature))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    pci = config.PCI['pci_value']
    pci_num = len(pci)
    if pci_only == True :
       model.add(Dense(pci_num))
    else :
       model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=16, shuffle=True)
    return model

#TREE
def build_DTReg_model(X_train, y_train):
    print "+++Decision Tree"
    model = MultiOutputRegressor(DecisionTreeRegressor(max_depth=30))
    model.fit(X_train, y_train)
    return model
def build_RFReg_model(X_train, y_train):
    n_estimators = 100
    print "+++Random Forest consisting of " + str(n_estimators) + " trees"
    #model = RandomForestRegressor(n_estimators=n_estimators, max_depth=30, random_state=2)
    model =  MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, max_depth=30, random_state=2))
    model.fit(X_train, y_train)
    return model
def build_GBTReg_model(X_train, y_train):
    n_estimators = 100
    print "+++GradientBoosting consisting of " + str(n_estimators) + " trees"
    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=n_estimators, max_depth=30, random_state=2))
    print X_train
    print y_train
    model.fit(X_train, y_train)
    return model

#KNN
def build_knn_model(X_train, y_train):
    print "+++KNN"
    model =  MultiOutputRegressor(KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto'))
    #model =  KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto')
    logging.info("model=%s" % model)
    model.fit(X_train, y_train)
    logging.info("X_train=%s" % X_train)
    logging.info("y_train=%s" % y_train)
    logging.info("model_fit=%s" % model.fit(X_train, y_train))
    #print "y_train==",y_train
    return model
