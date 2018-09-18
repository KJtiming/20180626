import numpy as np
import pandas as pd
import psutil as ps
import pickle as pkl
import os
import time
import sys
import model.utility
import model.learning_model
from model.utility import *
from model.learning_model import *
#from sklearn.metrics import accuracy_score 
import config
from keras.utils import plot_model
import seaborn as sns


if __name__ == '__main__':
    myProcess = ps.Process(os.getpid())
    begin_time = end_time = 0

    try:


        cmd = '****************************************\n'  \
              'chose training target ? \n' \
              '1. PCI \n' \
              '2. RSRP \n' \
              '3. RSRQ \n' \
              '4. SNR \n' \
              '****************************************\n'
        tr_tgt = raw_input(cmd)

        print '***load data...'
        (X_train, y_train, X_test, y_test, dataset_train) = load_data(tr_tgt)

        inputString = raw_input('Use before training data ? (y/n)')
        if (inputString[0] == 'y' or inputString[0] == 'Y') :
            print 'use the older test'
            with open("./output/train.pkl", "r") as f:
                X_train, y_train, X_test, y_test = pkl.load(f)
        else :
            print 'use the newest test'
            with open("./output/train.pkl", "w") as f:
                pkl.dump([X_train, y_train, X_test, y_test], f)
        print 'X_train'
        print X_train

        cmd = '****************************************\n'  \
              'chose machine learning method ? \n'\
              '1. DNN \n' \
              '2. Decision Tree \n'\
              '3. Random Forest \n' \
              '4. Gradient Decent \n' \
              '5. KNN \n' \
              '6. Draw measurement point on the picture  \n' \
              '****************************************\n' 
        ml = raw_input(cmd)
        print '***begin to train...'
        begin_time = time.time()
        name = ''
        if ml == '1' :
            print '***begin Dnn'
            name = 'dnn'
            if tr_tgt == '1' :
                nb_feature = 2
                pci_only = True
                model = build_dNN_model(X_train, y_train, nb_feature, pci_only)
            else : 
                nb_feature = 3
                pci_only = False
                model = build_dNN_model(X_train, y_train, nb_feature, pci_only)
            #model.save('dnn_model.h5') 


        elif ml == '2' :
            print '***begin Decision Tree'
            name = 'dt'
            model = build_DTReg_model(X_train, y_train)
        elif ml == '3' :
            print '***begin Random Forest'
            name = 'rf'
            model = build_RFReg_model(X_train, y_train)
        elif ml == '4' :
            print '***begin Gradient Decent'
            name = 'gbt'
            model = build_GBTReg_model(X_train, y_train)
        elif ml == '5' :
            print '***begin KNN'
            name = 'knn'
            model = build_knn_model(X_train, y_train)
            #plot_model(model,to_file='./output/model.png', show_shapes=True, show_layer_names=True)
        else :
            name = 'none'
        end_time = time.time()

        #data_test = np.genfromtxt('set4_test_1.csv', delimiter=',')
        filename = './output/finalized_model_' + tr_tgt +'-' +  name + '.sav'
        #joblib.dump(model, filename)
        if name != 'none' :
            if ml is not '1' :
                joblib.dump(model, filename)
            print "read X_test"
            X_test = np.genfromtxt('set4_half_X_test.csv', delimiter=',')
            #X_test = np.genfromtxt('data_300_train_mod_1_X_test.csv', delimiter=',')
            y_pred = model.predict(X_test)
            #np.savetxt('./output/y_pred.csv', y_pred, delimiter=',', fmt='%f')
            #np.savetxt('y_train.csv', y_train, delimiter=',', fmt='%f')
            #np.savetxt('y_test.csv', y_test, delimiter=',', fmt='%f')
            print "read y_test"
            y_test = np.genfromtxt('set4_half_y_test.csv', delimiter=',')
            #y_test = np.genfromtxt('data_300_train_mod_1_y_test.csv', delimiter=',')
            print "complete y_test"
            #RMSE_new = np.sqrt(np.square(np.subtract(y_test, y_pred)).mean())
            #print('RSRP RMSE_new: ' + str(RMSE_new))
            print "---%s Model--- %s features" %(('Multi-output' if is_multioutput else 'Single-output'), nb_feature)
            print "[MSE]: %.3f" % mean_squared_error(y_test, y_pred)
            print "[RMSE]: %.3f" % mean_squared_error(y_test, y_pred)**(0.5)
            #print '[R2]: %.3f' % r2_score(y_test, y_pred)
            #print '[AS]: %.3f' % accuracy_score(y_test, y_pred)
            print '[Training time] : %.3f' % (end_time - begin_time)
            print '[Memory] : %.3f' % (myProcess.memory_info()[0]/2.**20)  # RSS in MB
            #draw_importance_forest(model, nb_feature)
            #draw_heatmap(model, nb_feature, name)
            #draw_heatmap_pci(model, nb_feature, name)
            #draw_heatmap_single_pci(model, nb_feature, name)
            #draw_pci_bitmap(X_train)2

            #draw_bitmap(X_train) 
            if tr_tgt == '1' :
                #print "pci"
                draw_pci_heatmap(model, nb_feature, name, dataset_train, X_test, y_test, -10, 64, -5, 0)
            else :
                #draw_heatmap2(model, nb_feature, name)
                #draw_heatmap(model, nb_feature, name)
                scaler = True
                pci_enable = True
                draw_heatmap_generate( model, nb_feature, name, -10, 64, -5, 0, scaler, pci_enable)
            #draw_pci_bitmap(dataset_train)
        else :
            draw_pci_bitmap(dataset_train)

    except KeyboardInterrupt:
        model.save(fn_model)
