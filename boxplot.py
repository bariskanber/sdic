#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

import socket
MY_PC=1 if socket.gethostname()=="bkanber-gpu" else 0

matplotlib.rcParams['font.family'] = "Lohit Devanagari"
matplotlib.rcParams['font.size'] = 12

for run in range(0,100):
    os.system('scp comic:~/MNIST/results.csv ./')

    files=['results10k.csv','results30k.csv','results0pad.csv','results.csv']
    plt.figure(figsize=(12,6))
    sp=0
    for file in files:
        if os.path.exists(file):
            df=pd.read_csv(file,header=None, names=['method','loss','accuracy'])

            #df=df[df.method!='nn-entropy']
            print(len(df),len(df)/len(np.unique(df.method)))

            #df=df.loc[run*len(np.unique(df.method)):len(np.unique(df.method))*(run+1)-1]
            print(df)

            #df['error_rate']=(1-df.accuracy)*100

            #print(df)
            #print(df.columns.values)

            sp+=1
            plt.subplot(1,2,sp)
            sns.boxplot(y='accuracy',x='method',data=df)
            sp+=1
            plt.subplot(1,2,sp)
            sns.boxplot(y='loss',x='method',data=df)
            plt.title(str(len(df))+ " row(s)")
        # plt.ylim([0.5,3.5])

    if MY_PC: plt.show()