import sys
import config
from sklearn.externals import joblib 
from model.utility import *

if __name__ == '__main__':

    print sys.argv
    pwr1 = sys.argv[1]
    pwr2 = sys.argv[3]
    pattern1 = sys.argv[2]
    pattern2 = sys.argv[4]
    pci_enable = False
    '''
    a = isinstance(pwr1, str)
    print "a==",a
    pwr1=int(pwr1)
    b = isinstance(pwr1, float)
    print "b==",b
    pwr2=int(pwr2)
    '''
    generate_testing_input(pwr1, pwr2, pattern1, pattern2, pci_enable)

    pci_training_model = config.MODEL['path'] + config.MODEL['pci_model_name']
    rf_training_model = config.MODEL['path'] + config.MODEL['rf_model_name']


    pci_model = joblib.load(pci_training_model)
    cal_map_pci (pci_model, 2, pwr1, pwr2, pattern1, pattern2)

    rf_model = joblib.load(rf_training_model)
    scaler = False
    pci_enable = True
    draw_heatmap_generate(rf_model, 2, 'rf', pwr1, pattern1, pwr2, pattern2, scaler, pci_enable)


