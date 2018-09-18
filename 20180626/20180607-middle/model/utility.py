import numpy as np
import pandas as pd
import psutil as ps
import pickle as pkl
import os
import time
import math
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import config
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from PIL import Image
from keras.utils import plot_model
from pprint import pprint
import logging
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import seaborn as sns
import matplotlib.pyplot as plt2

logging.basicConfig(filename='ai_log.log',format='%(asctime)s  [%(levelname)s] : [%(filename)s:%(lineno)d] %(message)s', level=logging.DEBUG)
logging.debug('debug test')
logging.info('info test')
logging.warning('warning test')

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }


path = './data'
fn_heatmap_pixel = path + '/heatmap_pixel_1m.csv'
map_size = [104, 26]
nb_fold = 10


path2 = '../data/51-5F'
data_itri_51_5 = path + '/itri-51-5F.csv'

itri_data = path + '/51-5F/Data300.csv'


path = './data'
fn_train_data = path + '/train_pci.csv'
fn_test_data = path + '/MDT_testset.csv'
fn_model = 'model1'

train_data_size = 400
test_data_size = 12000
#train_data_size = 50000
#test_data_size = 10000
nb_epoch = 500
nb_feature = 2
is_multioutput = False  # (single-output target:serving Rx) / (multi-output target:serving Rx+neighbor Rx)
begin_time = end_time = 0

def load_pixel_data(nb_feature):
    dataset = pd.read_csv(fn_heatmap_pixel).values
    pixel = dataset[:, :nb_feature]
    return pixel

def load_training_pci_data():
    pci_data = './output/pci_result.csv'
    dataset = pd.read_csv(pci_data).values
    return dataset


def convert_location_data(x, y) :
    '''
    lng = a * x1 - b * y2 + c 
    lat = a * x2 + b * y1 + d

    NEMO -> indoor position value
    (840, -351) -> (96.824, 0)
    (923, -179) -> (107.068, 15.874)
    '''
    
    a = 0.1185
    b = -0.003
    c = -1.81
    d = 39.842
 
    lng = a * x - b * y * (-1) + c - 4
    lat = a * y * (-1) + b * x + d - 0.5
    
    return lng, lat
def convert_location_data_temp(x, y) :
    '''
    lng = a * x1 - b * y2 + c 
    lat = a * x2 + b * y1 + d

    NEMO -> indoor position value
    (840, -351) -> (96.824, 0)
    (923, -179) -> (107.068, 15.874)
    '''
    '''
    a = 0.1185
    b = -0.003
    c = -1.81
    d = 39.842
 
    lng = a * x - b * y * (-1) + c - 4
    lat = a * y * (-1) + b * x + d - 0.5
    '''
    lng = x
    lat = y
    return lng, lat


def tran_location_data (path, tr_tgt) :
    dataset = pd.read_csv(path).values
    pci_only = False
    result = []
    enb_feature_begin = config.GERNAL['enb_feature_begin']
    enb_feature_num = config.GERNAL['enb_feature_num']
    logging.debug('debug tran_location_data')
    if (tr_tgt == '1') :
        pci_only = True
    else :
        pci_only = False
    #np.savetxt('./output/test_dataset.csv', dataset, delimiter=',', fmt='%f')   
    for i in dataset :
        #for merge temp
        #lng, lat = convert_location_data_temp(i[0], i[1])
        lng, lat = convert_location_data(i[0], i[1])
        ue_feature = np.array([lng, lat])
        enb_feature = i[enb_feature_begin: enb_feature_begin + enb_feature_num]
        #a = isinstance(i[8], int)
        #print "a==",a
        data = np.concatenate((ue_feature, enb_feature), axis = 0)

        pci_data =  i[2: 2+1]  #put the pci value to the end
        #print "pci_data==",pci_data
        #data2 = data + pci_data
        #print data2
        #raw_input()

        data2 = np.concatenate((data, pci_data), axis = 0)
        #print "data2==",data2
        #result.append(data2)

        #put the target data to the end
        #num = int(tr_tgt) + 1]
        #if math.isnan(i[int(tr_tgt) + 1]) :
        #    continue
        if pci_only == False :
            if  i[int(tr_tgt) + 1] != '' and  not(math.isnan(i[int(tr_tgt) + 1])):
                target_data = i[int(tr_tgt) + 1 : int(tr_tgt) + 2]
                #data2 = np.concatenate((data2, target_data), axis = 0)
                feature_data = np.concatenate((data2, target_data), axis = 0)
        else :
            feature_data = data2
        #print "feature_data==",feature_data
        result.append(feature_data)
    
    '''

        if pci_only == True :
            ue_feature = np.array([lng, lat, i[2]])
        else :
            if  i[int(tr_tgt) + 1] != '' and  not(math.isnan(i[int(tr_tgt) + 1])):
                ue_feature = np.array([lng, lat, i[2], i[int(tr_tgt) + 1]])
        enb_feature = i[enb_feature_begin: enb_feature_begin + enb_feature_num]
        data = np.concatenate((ue_feature, enb_feature), axis = 0)
        result.append(data)
    '''
    output = np.array(result)
    np.savetxt('./output/test_output.csv', output, delimiter=',', fmt='%f')
    #print '##############'

    return output

def load_data(tr_tgt):
    tail = None if is_multioutput else -1
    itri_data = config.FILE['path'] + config.FILE['file_name']
    train_data_size = config.GERNAL['train_data_size']
    nb_feature = config.GERNAL['nb_feature']
    pci = config.PCI['pci_value']
    enb_feature_num = config.GERNAL['enb_feature_num']
    pci_num = len(pci)
    #print 'pci_num =',pci_num
    dataset_train = tran_location_data (itri_data, tr_tgt)
    np.savetxt('./output/test_dataset_train.csv', dataset_train, delimiter=',', fmt='%f')
    pci_data = True
    if (pci_data is True) :
        if tr_tgt is '1' :  #target is pci, pci_only = True
            print '11111111111111'
            dataset_train_pci = pci_data_reasign (dataset_train, True)
            #print "dataset_train_pci==",dataset_train_pci[-30:]
            #print "len",len(dataset_train_pci)
            #print "train_data_size==",train_data_size
            print "dataset_train_pci==",dataset_train_pci[:2]
            X = dataset_train_pci[:train_data_size, :nb_feature + enb_feature_num]
            #X = dataset_train_pci[:train_data_size, :nb_feature]
            y = dataset_train_pci[:train_data_size, pci_num*(-1):]
            #y = dataset_train_pci[:train_data_size, -1:]
        else :
            print '22222222222'
            print "rsrp_dataset_train",dataset_train[:2]
            X = dataset_train[:train_data_size, :nb_feature + 1 + enb_feature_num]
            print "X_rsrp===",X[:2]
            y = dataset_train[:train_data_size, -1:  ]
            print "y_rsrp===",y[:2]
    else :
        print '3333333333'
        X = dataset_train[:train_data_size, :nb_feature + enb_feature_num]
        y = dataset_train[:train_data_size, -1: ]

    print '=========================='
    np.savetxt('./output/test_X.csv', X, delimiter=',', fmt='%f')
#    for i in range(len(X)) :
#        print X[i]
    print '=========================='
    np.savetxt('./output/test_y.csv', y, delimiter=',', fmt='%f')
    #print 'number of input = ',len(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    np.savetxt('X_train.csv', X_train, delimiter=',', fmt='%f')
    np.savetxt('X_test.csv', X_test, delimiter=',', fmt='%f')
    #X_train = np.genfromtxt('set4_train.csv', delimiter=',')
    #y_test = np.genfromtxt('set4_test_1.csv', delimiter=',')
    return (X_train, y_train, X_test, y_test, dataset_train)


def draw_heatmap(model, nb_feature, name):
    plt.figure(0)
    x_resolution = map_size[0]
    y_resolution = map_size[1]
    xaxis = np.linspace(0., x_resolution, x_resolution)
    yaxis = np.linspace(0., y_resolution, y_resolution)
    x, y = np.meshgrid(xaxis, yaxis)
    pixel_pos = load_training_pci_data()
    pixel_pos = pixel_pos[:, :3] 
    z = model.predict(pixel_pos)
    z = np.reshape(z, (y_resolution, x_resolution))
    #img=mpimg.imread('./pic/51_5F-3.png')
    img = plt.imread("./pic/resize.png")
    plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]])
    #img.resize(y_resolution , x_resolution)
    #img.resize(254,947)
    #plt.contourf(x, y, z, 200, cmap='jet')
    #transform y axie because put on the picture
    #plt.contourf(x, y_resolution -y, z, 200, cmap='jet', alpha=0.15)
    plt.contourf(x, y_resolution -y, z, 200, cmap='jet', alpha=0.9)

    #plt.clim(-130,-50)

    plt.colorbar()
    pic = './output/heapmap_' + name
    imgplot = plt.imshow(img)
    plt.savefig(pic, dpi=200)
    #plt.show()

def draw_heatmap_generate(model, nb_feature, name, pwr1, pattern1, pwr2, pattern2, scaler, pci_enable):
    #print "/n model==",model
    #print "/n pwr1==",pwr1
    #print "/n pattern1==",pattern1
    #print "/n pwr2==",pwr2
    #print "/n pattern2==",pattern2
    print "11111"
    plt.figure()
    x_resolution = map_size[0]
    y_resolution = map_size[1]
    xaxis = np.linspace(0., x_resolution, x_resolution)
    yaxis = np.linspace(0., y_resolution, y_resolution)
    x, y = np.meshgrid(xaxis, yaxis)
    pixel_pos = generate_testing_input (pwr1, pwr2, pattern1, pattern2, pci_enable)

    enb_feature_num = config.GERNAL['enb_feature_num']
    #print "/npixel_pos_old==",pixel_pos[0:30]
    
    pixel_pos = pixel_pos[:, : int(enb_feature_num) + 3] # 3 is lng lat pci
    #print "pixel_pos_new==",pixel_pos[0:5]
    print "pixel_pos[0:2]==",pixel_pos[0:2]
    z = model.predict(pixel_pos)
    print "263z==",z
    #logging.info("z_old==%s" % z[:50])
    a=z.min()
    b=z.max()
    
    logging.info("a=%s" % a)
    logging.info("b=%s" % b)
    for n,i in enumerate(z) :
      #logging.info("in-----------" )
  
      if i<-130 :
          #logging.info("z[n]<-130_original==%s" % z[n])
          z[n]=-130
          #logging.info("z[n]_-130==%s" % z[n])
          #logging.info("n==%s" %n)
      
      if i>-80 :
          #logging.info("z[n]>-80_original==%s" % z[n])
          z[n]=-80
          #logging.info("z[n]_-80==%s" % z[n])
          #logging.info("n==%s" %n)

    #z[1]=-81
    #z[2]=-135
    
    z = np.reshape(z, (y_resolution, x_resolution))
    #z[:,90:107] = 0
    np.savetxt('./output/z_new.csv', z, delimiter=',', fmt='%f')
    
    img = plt.imread("./pic/resize.png")
    #plt.contourf(x, y_resolution -y , z, 200, cmap='jet', alpha=1)
    
    
    pic1 = './output/heapmap_' + name + '.png'
    '''
    if scaler is True :
        plt.colorbar()
    pic = './output/heapmap_' + name

    imgplot = plt.imshow(img)
    plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]]) 
    if scaler is True :
        #plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]])
        #plt.colorbar()
        plt.savefig(pic, dpi=200)
        
    else :
        plt.axis('off')
        plt.savefig(pic, bbox_inches='tight', pad_inches=0)
        pic = './output/heapmap_nnn'
        plt.savefig(pic, bbox_inches='tight', pad_inches=0, transparent=True)
        im = Image.open('./output/heapmap_rf.png')
        im_rotatenew = im.transpose(Image.FLIP_TOP_BOTTOM)
        im_rotatenew.save('./output/im_rotate.png')
        im = Image.open('./output/im_rotate.png')
        im_resize=im.resize((1000,260))
        im_resize.save('./output/im_resize.png')
        img = Image.open('./output/heapmap_rf.png')
        x = 385
        y = 5
        w = 138
        h = 122
        region = img.crop((x, y, x+w, y+h))
        region = region.convert("RGB")
        region.save("./crop.png")
    '''
    
    #draw_merge()
    pic = './output/heapmap_' + name
    imgplot = plt.imshow(img)
    #plt.savefig('out.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(pic, dpi=200)
    print "z==",z
    #y = np.array(z,dtype='float64')
    y = z.astype(np.float)
    print "y==",y
    heatmap_rsrp = plt.pcolor(y,cmap='jet',alpha=1)
    #print "heatmap_rsrp==",heatmap_rsrp
    #plt2.axis('equal')
    #plt2.show(heatmap_rsrp)
    plt.savefig('new_1.png')

    img = plt.imread("./pic/51_5F-3.png")
    plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]])
    plt.savefig('rsrp_map.png', dpi=200)
    img = Image.open('rsrp_map.png')
    x = 160
    y = 360
    w = 995
    h = 250
    region = img.crop((x, y, x+w, y+h))
    region = region.convert("RGBA")
    region.save('crop_rsrp_map.png')
    im = Image.open('51_5F-3.png')
    im1_resize=im.resize((946,235))
    im1_resize.save('im1_resize.png')
    im2_resize = Image.open('im1_resize.png')
    print "size_im==",im.size
    print "im_mode==",im.mode
    
    img = Image.open('crop_rsrp_map.png')
    x = 400
    y = 0
    w = 995
    h = 250
    region = img.crop((x, y, x+w, y+h))
    region = region.convert("RGBA")
    region.save("image_half.png")
 
    #im1_resize.paste(region,(360,0))
    im1_resize.save("image_crop_half.png")
    blended = Image.blend(im1_resize, im2_resize, alpha=0.4)
    blended.save('./output/new_rsrp.png')
    #image3 = img.resize((946,235))
    image3 = img.resize((946,235))
    print "size_image3==",image3.size
    print "mode_image3==",image3.mode
    image3.save("blend_cut.png")
    blended = Image.blend(im1_resize, image3, alpha=0.7)
    blended.save('./output/new_allrsrp.png')

def changeImageSize(maxWidth, 
                    maxHeight, 
                    image):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage

def draw_merge():
    
    im = Image.open('./pic/51_5F-3.png')
    print "size_im==",im.size
    print "im_mode==",im.mode
    im1_resize=im.resize((1000,260))
    print "size_im1_resize==",im1_resize.size
    print "im1_resize_mode==",im1_resize.mode
    #cropedIm=im.crop((0,1000,0,1000))
    img = Image.open('./output/heapmap_nnn.png')
    #img = Image.open('./output/test.PNG')
    #print "size_img==",img.size
    #print "img_mode==",img.mode

    
    x = 30
    y = 10
    w = 493
    h = 115
    region = img.crop((x, y, x+w, y+h))
    #region = region.convert("RGB")
    print "size_region==",region.size
    print "region_mode==",region.mode
    region.save("./output/blend_new.png")
    im2_resize=region.resize((1000,260))
    print "size_im2_resize==",im2_resize.size
    im2_resize.save("./output/blend_im2.png")
    
    #image1 = changeImageSize(1000, 260, img)
    #print "size_change==",image1.size
    #image1.save("./output/blend_chang.png")
    image3 = img.resize((1000,260))
    print "size_image3==",image3.size
    print "mode_image3==",image3.mode
    image3.save("./output/blend_cut.png")
    cropedIm=Image.open('./output/blend_new.png')
    im2_resize=cropedIm.resize((1000,260))
    
    #img3 = Image.alpha_composite(im1_resize, image3)
    #img3.save('./output/blended_img3.png')
    
    #right_bottom = (0, 0)
    #im1_resize.paste(image3, right_bottom, RGB)
    #im1_resize.save('./output/blended_cutmergepaste.png')
    
    blended = Image.blend(im1_resize, im2_resize, alpha=0.5)
    blended.save('./output/blended_cutmerge.png')
    #blended = Image.blend(im1_resize, im2_resize, alpha=0.5)
    #blended.save('./output/blended.png')
    
    im1_resize.paste(im2_resize,(0,0), im2_resize)

    copyIm=im.copy()
    cropedIm_copy=cropedIm.copy()
    threshold=100
    dist=5
    arr=np.array(np.asarray(cropedIm_copy))
    r,g,b,a=np.rollaxis(arr,axis=-1)    
    mask=((r>threshold)
      & (g>threshold)
      & (b>threshold)
      & (np.abs(r-g)<dist)
      & (np.abs(r-b)<dist)
      & (np.abs(g-b)<dist)
      )
    arr[mask,3]=0
    cropedIm_copy=Image.fromarray(arr,mode='RGBA')
    cropedIm_copy.save('./output/out.png')
    copyIm.paste(cropedIm_copy,(0,0))
    im.show()
    copyIm.save(r'./output/paste.png')
    copyIm.paste(cropedIm, (0, 0), cropedIm)
    
   

def extend_predict_value (pred_result): 
   
    output = [] 
    for i in (pred_result) :
        print i
        for j in range(0, 8) :
            output.append(i)
    print '%%%%%%%%%%%'
    return output 
    

def draw_heatmap2(model, nb_feature, name):
    plt.figure(0)
    x_resolution = map_size[0]
    y_resolution = map_size[1]
    xaxis = np.linspace(0., x_resolution, x_resolution)
    yaxis = np.linspace(0., y_resolution, y_resolution)
    x, y = np.meshgrid(xaxis, yaxis)
    pixel_pos = load_training_pci_data()
    pixel_pos = pixel_pos[:, :3]
    z = model.predict(pixel_pos)
    print '------------------'
    #print z
    #print z.shape
    output = extend_predict_value (z)
    print '------------------'
    z = np.reshape(output, (y_resolution * 10, x_resolution * 10))
    #img=mpimg.imread('./pic/51_5F-3.png')
    img = plt.imread("./pic/resize.png")
    plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]])
    #img.resize(y_resolution , x_resolution)
    #img.resize(254,947)
    #plt.contourf(x, y, z, 200, cmap='jet')
    #transform y axie because put on the picture
    #plt.contourf(x, y_resolution -y, z, 200, cmap='jet', alpha=0.15)
    plt.contourf(x, y_resolution -y, z, 200, cmap='jet', alpha=0.15)

    plt.colorbar()
    pic = './output/heapmap_' + name
    imgplot = plt.imshow(img)
    plt.savefig(pic, dpi=200)
    #plt.show()


def draw_bitmap(X_train):
    plt.figure(1)
    for i in range(len(X_train)):
        plt.plot(round(X_train[i][0]), round(X_train[i][1]), color='b', marker='2')  #shift x, y by 0.5 unit
    #plt.grid(which = 'minor', color='black', linestyle='-', linewidth=0.5)
    img = plt.imread("./pic/51_5F.jpg")
    plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Location of user report')
    plt.savefig('bitmap_pci.png', dpi=200)
    #plt.show()

def draw_pci_bitmap(X_train):
    plt.figure(1)
    marker_type = '3'

    for i in range(len(X_train)):
        if int(i) % 100 == 0 :
            print 'draw the i = ', i
        pci = int(X_train[i][2]) 
        '''
        if pci == 301 :
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='blue',   marker = marker_type) 
        elif pci == 302 :
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='green',  marker = marker_type )  
        elif pci == 120 :  
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='red',    marker = marker_type )  
        elif pci == 154:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='chocolate', marker = marker_type )  
        elif pci == 151:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='skyblue',  marker = marker_type )    
        elif pci == 448:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='yellow',  marker = marker_type ) 
        elif pci == 404:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='deeppink',  marker = marker_type ) 
        elif pci == 433:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='orange',  marker = marker_type ) 
        else:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='black',  marker = marker_type )  
        '''
        
        if pci == 37 :
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='blue',   marker = marker_type) 
        elif pci == 38 :
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='green',  marker = marker_type )  
        elif pci == 39 :  
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='red',    marker = marker_type )  
        elif pci == 40:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='skyblue', marker = marker_type )  
        elif pci == 41:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='orange',  marker = marker_type )
        elif pci == 42:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='deeppink',  marker = marker_type )
        '''
        if pci == 1 :
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='blue',   marker = marker_type) 
        elif pci == 2 :
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='green',  marker = marker_type )  
        elif pci == 3 :  
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='red',    marker = marker_type )  
        elif pci == 54:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='purple', marker = marker_type )  
        elif pci == 151:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='yellow',  marker = marker_type )
        else:
            plt.plot(round(abs(X_train[i][0])), round(abs(X_train[i][1])), color='black',  marker = marker_type )  
    '''
    '''
    #SC1 (206, 735)
    x, y = convert_location_data (868, 199)
    plt.plot((abs(x)), round(abs(y)), color='blue', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8 )

    #SC1 (199, 868)
    x, y = convert_location_data (735, 206)
    plt.plot(round(abs(x)), round(abs(y)), color='green', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)

    #SC1 (753, 152)
    x, y = convert_location_data (753, 152)
    plt.plot(round(abs(x)), round(abs(y)), color='red', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    
    #SC1 (631, 209)
    x, y = convert_location_data (631, 209)
    plt.plot(round(abs(x)), round(abs(y)), color='purple', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    '''
    
    x, y = convert_location_data (260,215) #37
    plt.plot(round(abs(x)), round(abs(y)), color='blue', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)

    #SC1 (199, 151)
    #x, y = convert_location_data (199, 868)
    x, y = convert_location_data (480, 158) #38
    plt.plot(round(abs(x)), round(abs(y)), color='green', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    
    x, y = convert_location_data (630, 210) #39
    #plt.plot(round(abs(x)), round(abs(y)), color='red', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    plt.plot(round(abs(x)), round(abs(y)), color='purple', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    
    x, y = convert_location_data (710, 275) #40
    plt.plot(round(abs(x)), round(abs(y)), color='skyblue', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    
    x, y = convert_location_data (765, 145) #41
    #plt.plot(round(abs(x)), round(abs(y)), color='orange', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    plt.plot(round(abs(x)), round(abs(y)), color='red', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)

    x, y = convert_location_data (908, 130) #42
    plt.plot(round(abs(x)), round(abs(y)), color='deeppink', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8) 
    #plt.grid(which = 'minor', color='black', linestyle='-', linewidth=0.5)
    img = plt.imread("./pic/51_5F-3.png")
    plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]])
    #plt.xlabel('x')
    #plt.ylabel('y')
    font['color'] = 'blue'
    plt.text(0, -8, 'PCI 301 is blue,',        fontdict=font)
    font['color'] = 'green'
    plt.text(30, -8, 'PCI 302 is green,',      fontdict=font)
    font['color'] = 'red'
    plt.text(65, -8, 'PCI 120 is red,',        fontdict=font)
    #font['color'] = 'purple'
    #plt.text(0, -12, 'PCI 54 is purple,',    fontdict=font)
    #font['color'] = 'black'
    #plt.text(30, -12, 'PCI 151 is black,',    fontdict=font)

#    font['color'] = 'black'
#    plt.text(60, -12, 'Unkonw PCI is black', fontdict=font)
    plt.title('Location of user report')
    plt.savefig('./output/pci.png', dpi=200)


def draw_heatmap_pci(model, nb_feature, name):
    plt.figure(0)
    x_resolution = map_size[0]
    y_resolution = map_size[1]
    xaxis = np.linspace(0., x_resolution, x_resolution+1)
    yaxis = np.linspace(0., y_resolution, y_resolution+1)
    x, y = np.meshgrid(xaxis, yaxis)
    pixel_pos = load_pixel_data(nb_feature)
    print pixel_pos
    output = model.predict(pixel_pos)

    pci = []
    #Get the maxium output
    #for i in range(len(output)) :
    #    print output[i]
    for i in output :
        i = i.astype(np.float)
        max_idx = np.argmax((i))
        if (i[max_idx] == 0.0) :
            pci.append(6)
        else :
            pci.append(max_idx)

    z = np.reshape(pci, (y_resolution+1, x_resolution+1))
    #print 'pci = ', z
    plt.contourf(x, y, z, 500, cmap='jet')
    plt.colorbar()
    plt.text(2, 0.65, r'pci = 54, pci = 3, pci = 2, pci = 1', fontdict=font)
    pic = './output/heapmap_pci_' + name
    plt.savefig(pic, dpi=200)
    #plt.show()



def cal_map_pci (model, nb_featurei, pwr1, pwr2, pattern1, pattern2) :

    #pixel_pos = load_pixel_data(nb_feature)
    pixel_pos = generate_pci_testing_input (pwr1, pwr2, pattern1, pattern2)
    enb_feature_num = config.GERNAL['enb_feature_num']
    print pixel_pos

    pixel_pos = pixel_pos[:, : int(enb_feature_num) + 2] # 3 is lng lat
    #print "/npixel_pos_old==",pixel_pos[0:30]

    output = model.predict(pixel_pos)
    #print "/npixel_pos_new==",pixel_pos[0:30]
    #plot_model(model,to_file='./output/model.png', show_shapes=True, show_layer_names=True)
    x_resolution = map_size[0]
    y_resolution = map_size[1]

    pci = []
    for i in output :
        i = i.astype(np.float)
        max_idx = np.argmax((i))
        if (i[max_idx] == 0.0) :
            pci.append(-1)  # all pci probability is 0
        else :
            pci.append(max_idx)
    z = np.reshape(pci, (y_resolution+1, x_resolution+1))

    f = open('./output/pci_result.csv', 'w') #write the pci result into file
    f.write('x,y,pci,\n')
    pci_input =  config.PCI['pci_value']
    pci_config = np.array(pci_input)


    for j in range(y_resolution):
        for i in range(x_resolution) :
            pci = z[j][i]
            pci_real = pci_config[pci]
            result = str(i) + ',' + str(j) + ',' + str(pci_real) +',' + '\n'
            f.write(result)  # python will convert \n to os.linesep
    f.close()



#latest
def draw_pci_heatmap(model, nb_feature, name, dataset_train, X_test, y_test, pwr1, pattern1, pwr2, pattern2):
    plt.figure(0)
    x_resolution = map_size[0]
    y_resolution = map_size[1]
    xaxis = np.linspace(0., x_resolution, x_resolution+1)
    yaxis = np.linspace(0., y_resolution, y_resolution+1)
    x, y = np.meshgrid(xaxis, yaxis)
    #pixel_pos = load_pixel_data(nb_feature)
    pixel_pos = generate_pci_testing_input(pwr1, pwr2, pattern1, pattern2)
    #print pixel_pos[:2]

    enb_feature_num = config.GERNAL['enb_feature_num']
    #print pixel_pos

    pixel_pos = pixel_pos[:, : int(enb_feature_num) + 2] # 3 is lng lat pci
    np.savetxt('pixel_pos.csv', pixel_pos, delimiter=',', fmt='%f')
    output = model.predict(pixel_pos)
    #print "672output==",output
    pci = []
    #Get the maxium output
    #for i in range(len(output)) :
    #    print output[i]
    #raw_input()
    for i in output :
        i = i.astype(np.float)
        max_idx = np.argmax((i))
        #print "max_idx==",max_idx
        if (i[max_idx] == 0.0) :
            pci.append(1)  # all pci probability is 0
        else :
            pci.append(max_idx)

    z = np.reshape(pci, (y_resolution+1, x_resolution+1))
    #print "688pci==",pci
    #print "689z==",z

    count_accurate(z, dataset_train, X_test, y_test )


    
    f = open('./output/pci_result.csv', 'w') #write the pci result into file
    f.write('x,y,pci,\n')
    pci_input =  config.PCI['pci_value']
    pci_config = np.array(pci_input)

    

##    for i in range(x_resolution):
##        for j in range(y_resolution) : 
    for j in range(y_resolution):
        for i in range(x_resolution) : 
          
            pci = z[j][i]
            #print "pci==",pci
            #print 'pci = ', pci
            
            if pci == 0 :#37
                pci_real = pci_config[0] 
                plt.plot(round(i), round(j), color='blue', marker = 's', markersize=5, alpha=.1)
            elif pci == 1 :#38
                pci_real = pci_config[1]
                plt.plot(round(i), round(j), color='green', marker = 's', markersize=5, alpha=.1 )
            elif pci == 2 :#39
                pci_real = pci_config[2]
                plt.plot(round(i), round(j), color='red', marker = 's', markersize=5, alpha=.1 )
            elif pci == 3:#40
                pci_real = pci_config[3]
                plt.plot(round(i), round(j), color='skyblue', marker = 's', markersize=5, alpha=.1)
            elif pci == 4:#41
                pci_real = pci_config[4]
                plt.plot(round(i), round(j), color='orange', marker = 's', markersize=5, alpha=.1)
            elif pci == 5:#42
                pci_real = pci_config[5]
                plt.plot(round(i), round(j), color='deeppink', marker = 's', markersize=5, alpha=.1)
            elif pci == 6:#120
                pci_real = pci_config[6]
                plt.plot(round(i), round(j), color='yellow', marker = 's', markersize=5, alpha=.1)
            elif pci == 7:#151
                pci_real = pci_config[7]
                plt.plot(round(i), round(j), color='yellow', marker = 's', markersize=5, alpha=.1)
            elif pci == 8:#154
                pci_real = pci_config[8]
                plt.plot(round(i), round(j), color='yellow', marker = 's', markersize=5, alpha=.1)
            else :                 
                pci_real = -1
                plt.plot(round(i), round(j), color='white', marker = 's', markersize=5, alpha=.1 )
            
            '''
            if pci == 0 :#37
                pci_real = pci_config[0] 
                plt.plot(round(i), round(j), color='blue', marker = 's', markersize=5, alpha=.1)
            elif pci == 1 :#38
                pci_real = pci_config[1]
                plt.plot(round(i), round(j), color='green', marker = 's', markersize=5, alpha=.1 )
            elif pci == 2 :#39
                pci_real = pci_config[2]
                plt.plot(round(i), round(j), color='red', marker = 's', markersize=5, alpha=.1 )
            elif pci == 3:#40
                pci_real = pci_config[3]
                plt.plot(round(i), round(j), color='purple', marker = 's', markersize=5, alpha=.1)
            elif pci == 4:#40
                pci_real = pci_config[4]
                plt.plot(round(i), round(j), color='yellow', marker = 's', markersize=5, alpha=.1)
            
            else :                 
                pci_real = -1
                plt.plot(round(i), round(j), color='white', marker = 's', markersize=5, alpha=.1 )
            '''
            result = str(i) + ',' + str(j) + ',' + str(pci_real) +',' + '\n'
            f.write(result)  # python will convert \n to os.linesep

    f.close()  # you can omit in most cases as the destructor will call it
    
    #plt.grid(which = 'minor', color='black', linestyle='-', linewidth=0.5)
    #SC1 (206, 735)
    #x, y = convert_location_data (206, 735)
    #original x, y = convert_location_data (868, 199) #301
    x, y = convert_location_data (260,215) #37
    plt.plot(round(abs(x)), round(abs(y)), color='blue', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)

    #SC1 (199, 151)
    #x, y = convert_location_data (199, 868)
    x, y = convert_location_data (480, 158) #38
    plt.plot(round(abs(x)), round(abs(y)), color='green', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    
    x, y = convert_location_data (630, 210) #39
    plt.plot(round(abs(x)), round(abs(y)), color='red', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    #plt.plot(round(abs(x)), round(abs(y)), color='purple', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    
    x, y = convert_location_data (710, 275) #40
    plt.plot(round(abs(x)), round(abs(y)), color='skyblue', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    
    x, y = convert_location_data (765, 145) #41
    plt.plot(round(abs(x)), round(abs(y)), color='orange', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    #plt.plot(round(abs(x)), round(abs(y)), color='red', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)

    x, y = convert_location_data (908, 130) #42
    plt.plot(round(abs(x)), round(abs(y)), color='deeppink', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)
    #SC1 (753, 152)
    #x, y = convert_location_data (753, 152)
    #plt.plot(round(abs(x)), round(abs(y)), color='red', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)

    #SC1 (631, 209)
    #x, y = convert_location_data (631, 209)
    #plt.plot(round(abs(x)), round(abs(y)), color='purple', marker = 'o', markersize=10, markeredgecolor = 'black', markeredgewidth = 0.8)

    
    img = plt.imread("./pic/51_5F-3.png")
    plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]])
    #plt.xlabel('x')
    #plt.ylabel('y')
    font['color'] = 'blue'
    plt.text(0, -8, 'PCI 301 is blue,', fontdict=font)
    font['color'] = 'green'
    plt.text(30, -8, 'PCI 302 is green,', fontdict=font)
    font['color'] = 'red'
    plt.text(60, -8, 'PCI 120 is red,', fontdict=font)
#    font['color'] = 'purple'
#    plt.text(0, -12, 'PCI 54 is purple,', fontdict=font)
#    font['color'] = 'black'
#    plt.text(30, -12, 'PCI 151 is black', fontdict=font)

    title = 'pci heatmap ' + name
    plt.title(str(title))
    des = './output/pci_heatmap_'+ name +'.png'
    plt.savefig(str(des), dpi=200)
    #img = Image.open('./output/pci_heatmap_knn.png')
    img = Image.open(des)
    im = Image.open("51_5F-3.png")
    x = 600
    y = 360
    w = 682
    h = 250
    region = img.crop((x, y, x+w, y+h))
    region = region.convert("RGBA")
    region.save("./output/pci_image_half.png")
    im1_resize=im.resize((946,235))
    im1_resize.save('im1_resize.png')
    im2_resize = Image.open('im1_resize.png')
    im1_resize.paste(region,(395,-8))
    im1_resize.save("image_crop_half.png")
    blended = Image.blend(im1_resize, im2_resize, alpha=0.08)
    blended.save('./output/new_pci.png')

def count_accurate (result, dataset_train, X_test, y_test) :
    
    total = 0
    correct = 0
    for i in range(len(dataset_train)):
        y = int(dataset_train[i][0])
        x = int(dataset_train[i][1])
        if y > int(map_size[0]) or x > int(map_size[1]) or y < 0 or x < 0:
            continue;
        #print 'x = ',x, ' y = ',y
        pci = int(result[x][y])
        total = total + 1
        #print 'pci = ', pci
        #print 'int(dataset_train[i][2])  = ', int(dataset_train[i][-1]) 
        '''
        if (pci == 0 and int(dataset_train[i][-1]) == 1) :
            correct = correct + 1
        elif (pci == 1 and int(dataset_train[i][-1]) == 2) :
            correct = correct + 1
        elif (pci == 2 and int(dataset_train[i][-1]) == 3) :
            correct = correct + 1
        elif (pci == 3 and int(dataset_train[i][-1]) == 54) :
            correct = correct + 1
        '''
        if (pci == 0 and int(dataset_train[i][-1]) == 37) :
            correct = correct + 1
        elif (pci == 1 and int(dataset_train[i][-1]) == 38) :
            correct = correct + 1
        elif (pci == 2 and int(dataset_train[i][-1]) == 39) :
            correct = correct + 1
        elif (pci == 3 and int(dataset_train[i][-1]) == 40) :
            correct = correct + 1
        elif (pci == 4 and int(dataset_train[i][-1]) == 41) :
            correct = correct + 1
        elif (pci == 5 and int(dataset_train[i][-1]) == 42) :
            correct = correct + 1
        elif (pci == 6 and int(dataset_train[i][-1]) == 120) :
            correct = correct + 1
        elif (pci == 7 and int(dataset_train[i][-1]) == 151) :
            correct = correct + 1
        elif (pci == 8 and int(dataset_train[i][-1]) == 154) :
            correct = correct + 1
        elif (pci == 9 and int(dataset_train[i][-1]) == 1) :
            correct = correct + 1
        elif (pci == 10 and int(dataset_train[i][-1]) == 62) :
            correct = correct + 1
        
    #raw_input()
    print 'PCI Total SET= ', total
    print 'PCI Correct SET= ', correct



    pci2 = [];
    #Get the maxium output
    #for i in range(len(output)) :
    #    print output[i]
    for i in y_test :
        i = i.astype(np.float)
        max_idx = np.argmax((i))
        if (i[max_idx] == 0.0) :
            pci2.append(6)
        else :
            pci2.append(max_idx)

    pci2 = np.array(pci2)
    total2 = 0
    correct2 = 0
    for i in range(len(X_test)) :
        y = int(X_test[i][0])
        x = int(X_test[i][1])
        pci3 = int(pci2[i])
        if y > int(map_size[0]) or x > int(map_size[1]) or y < 0 or x < 0:
            continue;

        map_pci = int(result[x][y])

        total2 = total2 + 1;
        #print 'pci3 = ', pci3 , 'map_pci = ', map_pci
        if (pci3 == map_pci) :
            correct2 = correct2 + 1
        elif (pci3 == map_pci) :
            correct2 = correct2 + 1
        elif (pci3 == map_pci) :
            correct2 = correct2 + 1
        elif (pci3 == map_pci) :
            correct2 = correct2 + 1


        
    print 'PCI Total Test Set = ', total2
    print 'PCI Total Test Correct = ', correct2


    

# format -> lng, lat, pci1, pci2, pci3, ..., rsrp/rsrq/snr
def pci_data_reasign (dataset_train, pci_only) :
    pci_input =  config.PCI['pci_value']
    enb_feature_begin = config.GERNAL['enb_feature_begin']
    enb_feature_num = config.GERNAL['enb_feature_num']
    pci = np.array(pci_input)
    dataset = []
    #print dataset_train
    for i in range( len(dataset_train)) :
        temp = []
        temp = dataset_train[i][0 : 2 + enb_feature_num].tolist()
        #print "temp",temp
        #print "/n dataset_train[i]==",dataset_train[i]
        tar = int (dataset_train[i][-1])
        #print "tar",tar
        #feature =  dataset_train[i][int(enb_feature_begin) : int(enb_feature_begin) + int(enb_feature_num)].tolist()
        for j in range (len(pci)) :
            #print 'tar = ', tar 
            #print 'pci_input[j] = ', pci_input[j] 
            if tar == int(pci_input[j]) :
                temp.append(1)
                #print "tar_1",tar
            else :
                temp.append(0)
        
        if pci_only is False :
            temp.append(int(dataset_train[i][ int(enb_feature_num) + 3: int(enb_feature_num) + 4][0])) # rsrp/rsrq/snr
            print dataset_train[i][ int(enb_feature_num) + 3: int(enb_feature_num) + 4][0]
            print 'MMMMMM'
        #else :
        dataset.append(temp)
        #print dataset_train[i]
        #print "dataset==",dataset[:10]
    return np.array(dataset)

def tran_pattern (pattern1, pattern2):
    print "/n pattern1===",pattern1
    print "/n pattern2===",pattern2
    pattern = []
    for i in range(0,5) :
        if int(pattern1) == 0 and i==0:
            pattern.append(1)
        elif int(pattern1) == 32 and i==1:
            pattern.append(1)
        elif int(pattern1) == 64 and i==2:
            pattern.append(1)
        elif int(pattern1) == 96 and i==3:
            pattern.append(1)
        elif int(pattern1) == 288 and i==1:
            pattern.append(1)
        elif int(pattern1) == 288 and i==2:
            pattern.append(1)
        else :
            pattern.append(0)
    for i in range(0,5) :
        if int(pattern2) == 0 and i==0:
            pattern.append(1)
        elif int(pattern2) == 32 and i==1:
            pattern.append(1)
        elif int(pattern2) == 64 and i==2:
            pattern.append(1)
        elif int(pattern2) == 96 and i==3:
            pattern.append(1)
        elif int(pattern2) == 288 and i==1:
            pattern.append(1)
        elif int(pattern2) == 288 and i==2:
            pattern.append(1)
        else :
            pattern.append(0)
    print "pattern==",pattern
    return pattern

def cal_distance_to_cell (x1, y1, x2, y2) :


    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist


def generate_testing_input (pwr1, pwr2, pattern1, pattern2, pci_enable) :
    data = load_training_pci_data()

    pattern = tran_pattern(pattern1, pattern2)
    x1, y1 = convert_location_data (868, 199)
    x2, y2 = convert_location_data (735, 206)
    a = isinstance(pwr1, str)
    #print "a==",a
   
    testing_data = []
    for i in range(len(data)) :
        temp = []
        if pci_enable is False :
            temp = data[i][0 : 2].tolist()
        else :
            temp = data[i][0 : 3].tolist() 
        #print "temp",temp
        #add power
        temp.append(int(pwr1))
        temp.append(int(pwr2))
        #add beampattern
        temp = temp + pattern
        #add distance
        dis1 = cal_distance_to_cell (x1, y1, data[i][0], data[i][1])
        temp.append(dis1)
        dis2 = cal_distance_to_cell (x2, y2, data[i][0], data[i][1])
        temp.append(dis2)
        #add angle
        angle1 = '0'
        temp.append(angle1)
        angle2 = '0'
        temp.append(angle2)
        
        
        testing_data.append(temp)
    output = np.array(testing_data)
    logging.info("output==%s" % output[:30])
    return output
        
def generate_pci_testing_input (pwr1, pwr2, pattern1, pattern2) :
    data = load_pixel_data(2)
    a = isinstance(pwr1, int)
    print "a_pci==",a
    pattern = tran_pattern(pattern1, pattern2)
    x1, y1 = convert_location_data (868, 199)
    x2, y2 = convert_location_data (735, 206)


    testing_data = []
    for i in range(len(data)) :
        temp = []
        temp = data[i][0 : 2].tolist()
        #add power
        temp.append(int(pwr1))
        temp.append(int(pwr2))
        #add beampattern
        temp = temp + pattern
        #add distance
        dis1 = cal_distance_to_cell (x1, y1, data[i][0], data[i][1])
        #temp.append(dis1)
        dis2 = cal_distance_to_cell (x2, y2, data[i][0], data[i][1])
        #temp.append(dis2)
        #add angle
        #angle1 = '0'
        #temp.append(angle1)
        #angle2 = '0'
        #temp.append(angle2)

        testing_data.append(temp)
    output = np.array(testing_data)
    return output
    





 
