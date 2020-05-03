#!/usr/bin/env python3

"""SDIC - Sparse matrix to structured imageset conversion 
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sdic import civic
import pandas as pd
import os

if os.path.exists('results.csv'): os.remove('results.csv')

import socket
MY_PC=1 if socket.gethostname()=="bkanber-gpu" else 0

if MY_PC:
    import matplotlib.pyplot as plt

MODE_ORIGINAL="asis"
MODE_RANDOM="rand"
MODE_LINEAR="sdic"
MODE_CIRCULAR="sdic_c"
MODE_DEEPINSIGHT="di"

CLASSIFIER_NN="cnn"
CLASSIFIER_RF="rf"
CLASSIFIER_LGM="lgm"

modes=[MODE_ORIGINAL,MODE_RANDOM,MODE_LINEAR,MODE_CIRCULAR,MODE_DEEPINSIGHT,CLASSIFIER_RF]
#modes=[CLASSIFIER_RF]

for run in range(0,50):
    for mode in modes:
        print("Operating mode: "+mode)
        if mode==CLASSIFIER_RF:
            mode=MODE_ORIGINAL
            classifier=CLASSIFIER_RF
        else:
            classifier=CLASSIFIER_NN
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
            np.random.seed(run)

            img_rows, img_cols = 12, 12
            img_size=12
            batch_size = 512
            num_classes = 2

            #http://archive.ics.uci.edu/ml/datasets/mushroom
            #https://dtai.cs.kuleuven.be/CP4IM/datasets/data/mushroom.txt
            #https://dtai.cs.kuleuven.be/CP4IM/datasets/
            x_train=np.zeros((8124,119+img_rows*img_cols-119))
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

                    for i in range(0,20):
                        x=np.random.randint(0,119)
                        x_train[row,x]=1-x_train[row,x]
                    row+=1

            x_train=x_train.reshape((x_train.shape[0],img_size,img_size))

            ind=list(range(0,x_train.shape[0]))
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
                vic.fit(x_train,None)
                x_train_new=vic.transform(x_train)
                x_test_new=vic.transform(x_test)
            elif mode==MODE_CIRCULAR:
                vic=civic("civic")
                vic.fit(x_train,None)
                x_train_new=vic.transform(x_train)
                x_test_new=vic.transform(x_test)
            else:
                raise Exception("Unknown operating mode")

            if (mode==MODE_LINEAR or 1) and run==0 and MY_PC:
                for j in range(0,5):
                    i=np.random.randint(0,x_train.shape[0])
                    print(np.sum(x_train[i]),np.sum(x_train_new_rand[i]),np.sum(x_train_new[i]))
                    plt.subplot(221),plt.imshow(x_train[i],cmap='gray')
                    plt.subplot(223),plt.imshow(x_train_new_rand[i],cmap='gray')
                    plt.subplot(224),plt.imshow(x_train_new[i],cmap='gray')
                    plt.title(mode)
                    plt.show()

            x_train=x_train_new
            x_test=x_test_new

        if 1:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
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
                if num_classes==2:
                    model = Sequential()
                    ks=(6,6)
                    k=(8+1)*4
                    model.add(Conv2D(k, kernel_size=(ks),
                                    activation='relu',input_shape=input_shape
                                    ))
                    model.add(Conv2D(k*2, ks, activation='relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(Dropout(0.3)) #0.30
                    model.add(Flatten())
                    model.add(Dense(64, activation='relu'))
                    model.add(Dropout(0.2)) #0.20
                    model.add(Dense(num_classes, activation='softmax'))
                else:
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
                p = model.predict(x_test)
                if num_classes==2:
                    auc=roc_auc_score(y_test[:,1],p[:,1])
                else:
                    auc=roc_auc_score(y_test,p,average='macro',multi_class='ovo')

                accs.append(score[1])
                losses.append(score[0])
                print('Mode:', mode, 'score:', score, 'AUC:', auc)
                
            with open('results.csv','at') as f:
                f.write('%s,%f,%f,%f\n'%(mode,np.mean(losses),auc,np.mean(accs)))
            #p=model.predict(x_test)
            #print('Test accuracy(sklearn):', accuracy_score(np.argmax(y_test,axis=1),np.argmax(p,axis=1)))
        elif classifier==CLASSIFIER_RF:
            Q=x_test.shape[0]//2
            x_val=x_test[:Q]
            y_val=y_test[:Q]
            x_test=x_test[Q:]
            y_test=y_test[Q:]

            x_train=x_train.reshape(x_train.shape[0],img_size*img_size)
            x_test=x_test.reshape(x_test.shape[0],img_size*img_size)

            from sklearn.ensemble import RandomForestClassifier
            if num_classes==2:
                clf=RandomForestClassifier(n_estimators=2000,verbose=1,criterion='entropy',n_jobs=10)
            else:
                clf=RandomForestClassifier(n_estimators=2000,verbose=1,criterion='entropy',n_jobs=10)
            print(clf)
            clf.fit(x_train,y_train)
            p=clf.predict_proba(x_test)
            if num_classes==2:
                loss=log_loss(y_test,p)
            else:
                loss=log_loss(y_test,p)
            acc=accuracy_score(y_test,np.argmax(p,axis=1))
            if num_classes==2:
                auc=roc_auc_score(y_test,p[:,1])
            else:
                auc=roc_auc_score(y_test,p,average='macro',multi_class='ovo')
            print('Mode:', mode)
            print('Test loss:', loss)
            print('Test AUC:', auc)    
            with open('results.csv','at') as f:
                f.write('rf,%f,%f,%f\n'%(loss,auc,acc))
