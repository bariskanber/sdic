"""Sparse dataset to structured imageset conversion 
"""

__author__ = "Baris Kanber"
__email__ = "b.kanber@ucl.ac.uk"
__version__ = "1.0.0"

import numpy as np

SDIC_TYPE_SDIC = "sdic"
SDIC_TYPE_SDIC_C = "sdic_c"

class sdic:
    def __init__(self,sdic_type):
        """
        sdic_type: the type of sparse dataset to structured imageset conversion to apply (SDIC_TYPE_*)
        """
        self.sdic_type=sdic_type
        assert(self.sdic_type in (SDIC_TYPE_SDIC,SDIC_TYPE_SDIC_C))

    def fit(self,datain):
        """
        datain: a numpy array of shape n_samples x n_features
        """
        datain=datain.reshape((datain.shape[0],-1))
        self.corrmatrix=np.corrcoef(datain.transpose())
        np.fill_diagonal(self.corrmatrix,np.nan)

    def transform(self,datain):
        """
        datain: a numpy array of shape n_samples x n_features
        """
        if self.sdic_type==SDIC_TYPE_SDIC_C:
            return self.transform_SDIC_C(datain)

        if self.sdic_type==SDIC_TYPE_SDIC:
            return self.transform_SDIC(datain)

    def transform_SDIC(self,datain):
        """
        datain: a numpy array of shape n_samples x n_features
        """
        datain=datain.reshape((datain.shape[0],-1))
        corrmatrix=np.array(self.corrmatrix,dtype=self.corrmatrix.dtype)

        img_size=int(np.ceil(np.sqrt(datain.shape[1])))
        if img_size%2!=0: img_size+=1

        dataout=np.zeros((datain.shape[0],img_size,img_size))

        cx=cy=0
        dx=np.unravel_index(np.nanargmax(corrmatrix),corrmatrix.shape)[0]

        dir=1
        dataout[:,cy,cx]=datain[:,dx]
        while True:
            corrmatrix[:,dx]=np.nan
            dxp=dx
            try:
                dx=np.nanargmax(corrmatrix[dx,:])
            except:
                break
            if np.all(np.isnan(corrmatrix[dx,:])): break
            corrmatrix[dxp,:]=np.nan
            if np.nanmax(corrmatrix[dx,:])<=0:
                dx=np.unravel_index(np.nanargmax(corrmatrix),corrmatrix.shape)[0]

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
            if cy==img_size: break

            dataout[:,cy,cx]=datain[:,dx]
        
        return dataout
            
    def transform_SDIC_C(self,datain):
        """
        datain: a numpy array of shape n_samples x n_features
        """
        datain=datain.reshape((datain.shape[0],-1))
        corrmatrix=np.array(self.corrmatrix,dtype=self.corrmatrix.dtype)

        img_size=int(np.ceil(np.sqrt(datain.shape[1])))
        if img_size%2!=0: img_size+=1

        dataout=np.zeros((datain.shape[0],img_size,img_size))

        cx=cy=img_size//2-1
        dx=np.unravel_index(np.nanargmax(corrmatrix),corrmatrix.shape)[0]

        dir=4
        lem=1
        lemi=1
        dataout[:,cy,cx]=datain[:,dx]
        while True:
            corrmatrix[:,dx]=np.nan
            dxp=dx
            try:
                dx=np.nanargmax(corrmatrix[dx,:])
            except: 
                break
            if np.all(np.isnan(corrmatrix[dx,:])): break
            corrmatrix[dxp,:]=np.nan
            if np.nanmax(corrmatrix[dx,:])<=0:
                 dx=np.unravel_index(np.nanargmax(corrmatrix),corrmatrix.shape)[0]

            if lemi>=lem-2 and dir==4:
                cy-=1
                cx-=1
                if lem>1: cx-=1
                lem+=2
                lemi=0
                dir=1
            else:
                while True:
                    if dir==1:
                        cy+=1
                        if lemi==lem:
                            dir=2
                            cy-=1
                            cx+=1
                            lemi=0
                    elif dir==2:
                        cx+=1
                        if lemi==lem-1:
                            dir=3
                            cx-=1
                            cy-=1
                            lemi=0
                    elif dir==3:
                        cy-=1
                        if lemi==lem-1:
                            dir=4
                            cy+=1
                            cx-=1
                            lemi=0
                    elif dir==4:
                        cx-=1
                    if cx>=0 and cy>=0 and cx<img_size and cy<img_size:
                        break
                    else:
                        lemi+=1

            lemi+=1
            dataout[:,cy,cx]=datain[:,dx]

        return dataout
