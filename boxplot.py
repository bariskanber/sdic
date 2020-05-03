#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

sns.set(style="whitegrid")

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
            df=pd.read_csv(file,header=None, names=['method','loss','AUC','accuracy'])
            df.method='$'+df.method.astype(str)+'$'

            #df=df[df.method!='rf']
            print(len(df),len(df)/len(np.unique(df.method)))

            #df=df.loc[run*len(np.unique(df.method)):len(np.unique(df.method))*(run+1)-1]
            print(df)

            #df['error_rate']=(1-df.accuracy)*100

            #print(df)
            #print(df.columns.values)

            arglowestloss=np.argmin(df.loss.values)
            arghighestacc=np.argmax(df.accuracy.values)
            arghighestAUC=np.argmax(df.AUC.values)

            sp+=1
            plt.subplot(1,3,sp)
            sns.boxplot(y='accuracy',x='method',data=df,palette="Set3")
            sns.swarmplot(y='accuracy',x='method',data=df,color=".25")
            plt.title('%d,%d,%s'%(arglowestloss,arghighestacc,arghighestAUC))
            sp+=1
            plt.subplot(1,3,sp)
            sns.boxplot(y='AUC',x='method',data=df,palette="Set3")
            sns.swarmplot(y='AUC',x='method',data=df,color=".25")
            plt.title(str(len(df))+ " row(s)")
            sp+=1
            plt.subplot(1,3,sp)
            sns.boxplot(y='loss',x='method',data=df,palette="Set3")
            sns.swarmplot(y='loss',x='method',data=df,color=".25")
            plt.title(str(len(df))+ " row(s)")

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.30, hspace=0.55)
        # plt.ylim([0.5,3.5])

    if MY_PC: plt.show()