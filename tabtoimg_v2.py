#!/usr/bin/env python3

"""Tabular data to image conversion (TDIC) introducing the covariance informed TDIC
"""

__author__ = "Baris Kanber"
__email__ = "b.kanber@ucl.ac.uk"

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, GaussianNoise, ZeroPadding2D, Input, concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras import backend as K
import numpy as np
import lightgbm as lgb 
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from civic import civic
import pandas as pd
import os

if os.path.exists('results.csv'):
    os.remove('results.csv')

import socket
MY_PC=1 if socket.gethostname()=="bkanber-gpu" else 0

if MY_PC:
    import matplotlib.pyplot as plt

MODE_ORIGINAL="original"
MODE_RANDOM="random"
MODE_LINEAR="linear_v2"
MODE_CIRCULAR="circular"
MODE_ENTROPY="entropy"
MODE_DEEPINSIGHT="deepinsight"
MODE_YORDER="yorder"
MODE_OPT="opt"

CLASSIFIER_NN="nn"
CLASSIFIER_RF="rf"
CLASSIFIER_LGM="lgm"

classifier=CLASSIFIER_NN

#modes=[MODE_LINEAR,MODE_RANDOM,MODE_ORIGINAL] if classifier==CLASSIFIER_NN else [MODE_RANDOM]
modes=[MODE_LINEAR,MODE_CIRCULAR,MODE_RANDOM,MODE_ORIGINAL,MODE_DEEPINSIGHT] if classifier==CLASSIFIER_NN else [MODE_RANDOM]

for run in range(0,100):
    for mode in modes:
        print("Operating mode: "+mode)
        epochs = 9999
        show_plots=1

        if 1:
            img_rows, img_cols = 28,28
            img_size = 28
            batch_size = 256
            num_classes = 10
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            if 1:
                x_train=np.concatenate((x_train,x_test),axis=0)
                y_train=np.concatenate((y_train,y_test),axis=0)

                #x_train=x_train[y_train==0]
                #y_train=y_train[y_train==0]

                ind=list(range(0,x_train.shape[0]))
                np.random.seed(run)
                np.random.shuffle(ind)

                n_test=10000

                x_test=x_train[ind[-n_test:]]
                y_test=y_train[ind[-n_test:]]

                x_train=x_train[ind[:-n_test]]
                y_train=y_train[ind[:-n_test]]
        else:
            #http://archive.ics.uci.edu/ml/datasets/mushroom
            #https://dtai.cs.kuleuven.be/CP4IM/datasets/data/mushroom.txt
            #https://dtai.cs.kuleuven.be/CP4IM/datasets/
            x_train=np.zeros((8124,119+25))
            y_train=np.zeros((8124,1))
            with open('mushroom.txt') as f:
                row=0
                while True:
                    s=f.readline().replace('\n','')
                    if s is None or len(s)<1: break
                    tokens=s.split(' ')
                    for i in range(0,21):
                        x_train[row,int(tokens[i])]=1
                    y_train[row]=int(tokens[21])

                    for i in range(0,10):
                        x=np.random.randint(0,119)
                        x_train[row,x]=1-x_train[row,x]
                    row+=1

            img_rows, img_cols = 12, 12
            img_size=12
            batch_size = 512
            num_classes = 2

            x_train=x_train.reshape((x_train.shape[0],img_size,img_size))

            ind=list(range(0,x_train.shape[0]))
            np.random.seed(run)
            np.random.shuffle(ind)

            n_test=4000

            x_test=x_train[ind[-n_test:]]
            y_test=y_train[ind[-n_test:]]

            x_train=x_train[ind[:-n_test]]
            y_train=y_train[ind[:-n_test]]

        if mode!=MODE_ORIGINAL:
            x_train_new=np.zeros((x_train.shape[0],img_size,img_size))
            x_test_new=np.zeros((x_test.shape[0],img_size,img_size))

            if 1:
                import bz2
                import pickle
                import os
                x=np.zeros((img_size*img_size,x_train.shape[0]))
                xt=np.zeros((img_size*img_size,x_test.shape[0]))

                for i in range(0,x_train.shape[0]):
                    x[:,i]=np.ravel(x_train[i])

                for i in range(0,x_test.shape[0]):
                    xt[:,i]=np.ravel(x_test[i])

                if not os.path.exists('entropies.pickle'):
                    entropies=np.zeros((img_size*img_size,1))

                    for i in range(0,img_size*img_size):
                        print(i)
                        with bz2.BZ2File('smallerfile','wb') as f:
                            pickle.dump(x[i,:],f)
                        entropies[i]=os.stat('smallerfile').st_size

                    with open('entropies.pickle','wb') as f:
                        pickle.dump(entropies,f)
                else:
                    with open('entropies.pickle','rb') as f:
                        entropies=pickle.load(f)

                ind=np.argsort(entropies.flatten())

                x_train_new_ent=np.zeros((x_train.shape[0],img_size*img_size))
                x_test_new_ent=np.zeros((x_test.shape[0],img_size*img_size))
                for i in range(0,len(ind)):
                    x_train_new_ent[:,i]=x[ind[len(ind)-1-i],:]
                    x_test_new_ent[:,i]=xt[ind[len(ind)-1-i],:]

                x_train_new_ent=np.reshape(x_train_new_ent,(x_train_new.shape[0],img_size,img_size))
                x_test_new_ent=np.reshape(x_test_new_ent,(x_test_new.shape[0],img_size,img_size))

            if 1:
                x_train_new_rand=np.zeros((x_train.shape[0],img_size,img_size))
                x_test_new_rand=np.zeros((x_test.shape[0],img_size,img_size))
                ind=list(range(0,img_size*img_size))
                np.random.seed(run) #xxx
                np.random.shuffle(ind)
                cx=cy=0
                dir=1
                for i in range(0,img_size*img_size):
                    dy=ind[i]//img_size
                    dx=ind[i]%img_size
                    x_train_new_rand[:,cy,cx]=x_train[:,dy,dx] 
                    x_test_new_rand[:,cy,cx]=x_test[:,dy,dx] 
                    if dir==1: cx+=1
                    else: cx-=1
                    if cx==img_size:
                        cy+=1
                        cx-=1
                        dir*=-1
                    elif cx==-1:
                        cy+=1
                        cx+=1
                        dir*=-1

            if mode==MODE_DEEPINSIGHT:
                from sklearn.decomposition import KernelPCA
                pca=KernelPCA(n_components=2)
                X=x_train.reshape(x_train.shape[0],img_size*img_size)
                Xt=x_test.reshape(x_test.shape[0],img_size*img_size)

                x=pca.fit_transform(np.transpose(X))
                x[:,0]=x[:,0]-np.min(x[:,0])
                x[:,0]=x[:,0]/np.max(x[:,0])*(img_size-1)
                x[:,1]=x[:,1]-np.min(x[:,1])
                x[:,1]=x[:,1]/np.max(x[:,1])*(img_size-1)
                x=x.round().astype('int')

                pts_per_coord={}
                for i in range(0,x.shape[0]):
                    coord=(x[i,0],x[i,1])
                    x_train_new[:,x[i,0],x[i,1]]+=X[:,i]
                    if coord not in pts_per_coord:
                        pts_per_coord[coord]=1
                    else:
                        pts_per_coord[coord]+=1
                for coord in pts_per_coord:
                    x_train_new[:,coord[0],coord[1]]/=pts_per_coord[coord]

                pts_per_coord={}
                for i in range(0,x.shape[0]):
                    coord=(x[i,0],x[i,1])
                    x_test_new[:,x[i,0],x[i,1]]+=Xt[:,i]
                    if coord not in pts_per_coord:
                        pts_per_coord[coord]=1
                    else:
                        pts_per_coord[coord]+=1
                for coord in pts_per_coord:
                    x_test_new[:,coord[0],coord[1]]/=pts_per_coord[coord]

            elif mode==MODE_RANDOM:
                x_train_new=x_train_new_rand
                x_test_new=x_test_new_rand
            elif mode==MODE_LINEAR:
                vic=civic("civic_linear")
                vic.fit(x_train,entropies)
                x_train_new=vic.transform(x_train)
                x_test_new=vic.transform(x_test)
            elif mode==MODE_CIRCULAR:
                vic=civic("civic")
                vic.fit(x_train,entropies)
                x_train_new=vic.transform(x_train)
                x_test_new=vic.transform(x_test)
            elif mode==MODE_ENTROPY:
                x_train_new=x_train_new_ent
                x_test_new=x_test_new_ent
            elif mode==MODE_OPT:
                x_train_new=np.zeros((x_train.shape[0],img_size,img_size))
                x_test_new=np.zeros((x_test.shape[0],img_size,img_size))
                #ind=list(range(0,img_size*img_size))
                
                ind=[134, 200, 20, 30, 368, 205, 324, 146, 231, 236, 0, 299, 232, 261, 86, 358, 178, 220, 213, 241, 265, 224, 361, 163, 201, 346, 305, 335, 141, 17, 19, 109, 47, 116, 274, 12, 394, 219, 75, 338, 390, 39, 67, 196, 171, 357, 46, 399, 130, 252, 33, 53, 355, 362, 248, 119, 68, 296, 284, 302, 127, 187, 1, 226, 392, 65, 312, 7, 345, 34, 93, 221, 81, 176, 55, 15, 206, 371, 100, 193, 96, 262, 293, 255, 257, 278, 153, 170, 245, 120, 214, 259, 234, 11, 143, 364, 331, 88, 76, 212, 98, 92, 179, 328, 254, 340, 165, 387, 389, 228, 95, 29, 28, 396, 174, 304, 102, 72, 107, 6, 288, 323, 112, 334, 347, 44, 151, 292, 59, 104, 140, 56, 376, 150, 63, 344, 78, 352, 31, 90, 115, 154, 161, 138, 38, 264, 190, 295, 22, 218, 157, 89, 198, 27, 297, 385, 314, 149, 94, 266, 276, 388, 251, 365, 183, 243, 367, 313, 327, 247, 227, 303, 282, 326, 123, 5, 308, 320, 43, 103, 24, 48, 375, 159, 111, 209, 106, 136, 351, 283, 85, 279, 152, 321, 21, 42, 398, 258, 273, 80, 162, 301, 235, 330, 58, 225, 354, 285, 139, 356, 333, 36, 215, 242, 167, 145, 54, 207, 300, 377, 244, 393, 186, 290, 382, 306, 117, 184, 203, 45, 373, 74, 133, 3, 359, 199, 317, 18, 166, 322, 342, 353, 329, 291, 349, 395, 82, 8, 275, 147, 192, 210, 381, 315, 79, 101, 318, 97, 84, 386, 105, 35, 250, 191, 2, 194, 202, 222, 217, 270, 260, 177, 155, 156, 185, 332, 370, 182, 237, 144, 369, 4, 49, 108, 73, 135, 14, 52, 310, 267, 169, 66, 13, 269, 71, 383, 287, 91, 363, 230, 277, 70, 256, 216, 51, 10, 60, 148, 57, 168, 77, 246, 372, 348, 114, 341, 271, 391, 268, 311, 294, 50, 128, 379, 249, 374, 350, 173, 142, 239, 360, 175, 32, 132, 286, 125, 160, 129, 62, 343, 189, 26, 233, 298, 195, 121, 397, 181, 272, 37, 9, 223, 289, 280, 240, 204, 229, 253, 337, 316, 40, 208, 122, 25, 339, 238, 380, 87, 126, 16, 336, 83, 124, 64, 113, 281, 197, 137, 307, 61, 263, 69, 99, 23, 366, 188, 378, 131, 180, 211, 118, 319, 158, 172, 164, 41, 110, 384, 325, 309]
                #ind=ind[::-1]
                cx=cy=0
                dir=1
                for i in range(0,img_size*img_size):
                    dy=ind[i]//img_size
                    dx=ind[i]%img_size
                    x_train_new[:,cy,cx]=x_train[:,dy,dx] 
                    x_test_new[:,cy,cx]=x_test[:,dy,dx] 
                    if dir==1: cx+=1
                    else: cx-=1
                    if cx==img_size:
                        cy+=1
                        cx-=1
                        dir*=-1
                    elif cx==-1:
                        cy+=1
                        cx+=1
                        dir*=-1

                possum=np.mean(x_train_new[y_train==1],axis=0)
                negsum=np.mean(x_train_new[y_train==0],axis=0)
                print('ok')
            else:
                raise Exception("Unknown operating mode")

            if (mode==MODE_LINEAR or 1) and run==0 and MY_PC:
                for j in range(0,5):
                    i=np.random.randint(0,x_train.shape[0])
                    print(np.sum(x_train[i]),np.sum(x_train_new_ent[i]),np.sum(x_train_new_rand[i]),np.sum(x_train_new[i]))
                    plt.subplot(221),plt.imshow(x_train[i],cmap='gray')
                    plt.subplot(222),plt.imshow(x_train_new_ent[i],cmap='gray')
                    plt.subplot(223),plt.imshow(x_train_new_rand[i],cmap='gray')
                    plt.subplot(224),plt.imshow(x_train_new[i],cmap='gray')
                    plt.title(mode)
                    plt.show()

            x_train=x_train_new
            x_test=x_test_new

        # RL
        if 0:
            x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
            x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
            rlpos=[]
            rlneg=[]
            for samplei in range(0,x_train.shape[0]):
                rl=0
                for j in range(0,img_rows * img_cols):
                    if x_train[samplei,j]>0:
                        rl+=1
                    elif rl>0:
                        if y_train[samplei]==1: rlpos.append(rl)
                        else: rlneg.append(rl)
                        rl=0

            print('RL:',np.mean(rlpos),np.mean(rlneg))
            plt.boxplot([rlpos,rlneg]);plt.show()

        if 1:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            #x_train_new = x_train_new.reshape(x_train_new.shape[0], img_rows, img_cols, 1)
            #x_test_new = x_test_new.reshape(x_test_new.shape[0], img_rows, img_cols, 1)
            #x_train=np.concatenate((x_train,x_train_new),axis=3)
            #x_test=np.concatenate((x_test,x_test_new),axis=3)
            input_shape = (img_rows, img_cols,x_train.shape[3])

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        if classifier==CLASSIFIER_NN:
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            if np.max(x_train)>1:
                x_train /= 255
                x_test /= 255

            if 0:
                for ind in range(0,x_train.shape[0]):
                    if np.random.randint(0,10)==0:
                        XXX=np.expand_dims(np.rot90(x_train[ind]),axis=0)
                        x_train=np.concatenate((x_train,XXX),axis=0)
                        XXX=np.expand_dims(y_train[ind],axis=0)
                        y_train=np.concatenate((y_train,XXX),axis=0)
                    if np.random.randint(0,10)==1:
                        XXX=np.expand_dims(np.rot90(np.rot90(x_train[ind])),axis=0)
                        x_train=np.concatenate((x_train,XXX),axis=0)
                        XXX=np.expand_dims(y_train[ind],axis=0)
                        y_train=np.concatenate((y_train,XXX),axis=0)
                    if np.random.randint(0,10)==2:
                        XXX=np.expand_dims(np.fliplr(x_train[ind]),axis=0)
                        x_train=np.concatenate((x_train,XXX),axis=0)
                        XXX=np.expand_dims(y_train[ind],axis=0)
                        y_train=np.concatenate((y_train,XXX),axis=0)
                    if np.random.randint(0,10)==3:
                        XXX=np.expand_dims(np.flipud(x_train[ind]),axis=0)
                        x_train=np.concatenate((x_train,XXX),axis=0)
                        XXX=np.expand_dims(y_train[ind],axis=0)
                        y_train=np.concatenate((y_train,XXX),axis=0)
                print('x_train shape:', x_train.shape)
                ind=list(range(0,x_train.shape[0]))
                np.random.shuffle(ind)
                x_train=x_train[ind]
                y_train=y_train[ind]

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            callbacks = [
                        EarlyStopping(monitor='val_loss', patience=20 if num_classes==10 else 20, verbose=0),
                        ModelCheckpoint('model.hdf5', 
                            monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
                         ]

            Q=x_test.shape[0]//2
            x_val=x_test[:Q]
            y_val=y_test[:Q]
            x_test=x_test[Q:]
            y_test=y_test[Q:]

            accs=[]
            losses=[]

            for nnrun in range(0,1):
                if 1:
                    model = Sequential()
                    ks=(3,3)
                    model.add(Conv2D(32, kernel_size=(ks),
                                    activation='relu',input_shape=input_shape
                                    ))
                    model.add(Conv2D(64, ks, activation='relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(Dropout(0.25))
                    model.add(Flatten())
                    model.add(Dense(128, activation='relu'))
                    model.add(Dropout(0.5))
                    model.add(Dense(num_classes, activation='softmax'))
                elif 1:
                    model = Sequential()
                    ks=(3,3)
                    model.add(Conv2D(32, kernel_size=(ks),
                                    activation='relu',input_shape=input_shape
                                    ))
                    model.add(Conv2D(64, ks, activation='relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(Conv2D(64, ks, activation='relu'))
    #                model.add(MaxPooling2D(pool_size=(2, 2)))
    #                model.add(Conv2D(64, (1,1), activation='relu'))
                    model.add(Dropout(0.25))
                    model.add(Flatten())
                    model.add(Dense(128, activation='relu'))
                    model.add(Dropout(0.5))
                    model.add(Dense(num_classes, activation='softmax'))
                else:
                    ks=(1,1) if mode==MODE_ORIGINAL else (1,1)

                    input1 = Input(input_shape)
                    p1=Conv2D(32, kernel_size=(ks),
                                    activation='relu')(input1)
                    
                    p1=Conv2D(64, ks, activation='relu')(p1)
                    l1=MaxPooling2D(pool_size=(2, 2))(p1)
                    l1=Dropout(0.25)(l1)
                    l1=Flatten()(l1)
                    l1=Dense(2, activation='relu')(l1)

                    l2=MaxPooling2D(pool_size=(2, 2))(p1)
                    l2=MaxPooling2D(pool_size=(2, 2))(l2)
                    l2=Conv2D(64, ks, activation='relu')(l2)
                    l2=Dropout(0.25)(l2)
                    l2=Flatten()(l2)
                    l2=Dense(8, activation='relu')(l2)

                    merge1 = keras.layers.concatenate([l1,l2])

                    conv1=Dropout(0.5)(merge1)
                    output1=Dense(num_classes, activation='softmax')(conv1)

                    model = Model(input = input1, output = output1)

                model.compile(loss=keras.losses.categorical_crossentropy if num_classes>2 else keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=['accuracy'])

                print(model.summary())

                model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val))

                model=load_model('model.hdf5')

                score = model.evaluate(x_test, y_test, verbose=0)
                accs.append(score[1])
                losses.append(score[0])
                print('Mode:', mode, 'score:', score)
                
            with open('results.csv','at') as f:
                f.write('nn-%s,%f,%f\n'%(mode,np.mean(losses),np.mean(accs)))
            #p=model.predict(x_test)
            #print('Test accuracy(sklearn):', accuracy_score(np.argmax(y_test,axis=1),np.argmax(p,axis=1)))
        elif classifier==CLASSIFIER_LGM:
            max_depth=run+7
            print('max_depth',max_depth)
            lgbm_params =  {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    #'boost_from_average':False,
                    #'min_gain_to_split':1e-4, #default: 0
                    #'min_data_in_leaf':50, #default: 20
                    #'min_data_in_leaf':5, #default: 20
                    #'min_sum_hessian_in_leaf':1e-4, #default: 1e-3
                    #'min_gain_to_split':0, #default: 0
                    #'max_bin':255, #default: 255
                    #'min_data_in_bin':3, #default = 3
                    #'bin_construct_sample_cnt':200000*10, #default = 200000
                    #'scale_pos_weight':18871296.0/3072, #default = 1.0
                    'max_depth': max_depth,
                    #'max_depth': 14,
                    'num_leaves': 2^(max_depth),
                    #'feature_fraction': 0.80,
                    #'feature_fraction_seed':np.random.randint(0,1000000),
                    #'bagging_fraction': 0.90,
                    # 'bagging_freq': 5,
                    #'learning_rate': 0.001*(run+1),
                    #'lambda_l2': 5,
                    'num_threads':12 if True else 24,
                    'num_class':num_classes,
                }  

            x_train=x_train.reshape(x_train.shape[0],img_size*img_size)
            x_test=x_test.reshape(x_test.shape[0],img_size*img_size)

            lgtrain = lgb.Dataset(x_train, y_train)
            lgvalid = lgb.Dataset(x_test, y_test)
            
            try:
                lgb_clf = lgb.train(
                        lgbm_params,
                        lgtrain,
                        num_boost_round=10000,
                        valid_sets=[lgtrain, lgvalid],
                        valid_names=['train','valid'],
                        early_stopping_rounds=50,
                        verbose_eval=10
                    )
                best_iteration=lgb_clf.best_iteration
                p=lgb_clf.predict(x_test, num_iteration=best_iteration)

                print('Mode:', mode)
                print('Test loss:', log_loss(y_test,p))
                print('Test accuracy:', accuracy_score(y_test,np.argmax(p,axis=1)))
                with open('results.csv','at') as f:
                    f.write('lgm,%f,%f\n'%(accuracy_score(y_test,np.argmax(p,axis=1)),max_depth))
            except:
                pass
        elif classifier==CLASSIFIER_RF:
            x_train=x_train.reshape(x_train.shape[0],img_size*img_size)
            x_test=x_test.reshape(x_test.shape[0],img_size*img_size)

            n_estimators=(run+1)*10

            from sklearn.ensemble import RandomForestClassifier
            clf=RandomForestClassifier(n_estimators=n_estimators,min_samples_leaf=1)
            clf.fit(x_train,y_train)
            p=clf.predict_proba(x_test)
            print('Mode:', mode)
            #print('Test loss:', log_loss(y_test,p,labels=range(0,num_classes-1)))
            print('Test accuracy:', accuracy_score(y_test,np.argmax(p,axis=1)))    
            with open('results.csv','at') as f:
                f.write('rf,%f,%f\n'%(accuracy_score(y_test,np.argmax(p,axis=1)),n_estimators))
