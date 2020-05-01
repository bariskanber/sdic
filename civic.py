import numpy as np

class civic:
    VIC_TYPE_CIVIC="civic"
    VIC_TYPE_CIVIC_LINEAR="civic_linear"
    
    def __init__(self,vic_type):
        self.vic_type=vic_type
        self.reverse=False

    def fit(self,datain,entropies):
        """
        datain: a numpy array of shape n_samples x n_features
        """
        datain=datain.reshape((datain.shape[0],-1))
#        self.covmatrix=np.cov(datain.transpose())
        self.covmatrix=np.corrcoef(datain.transpose())
        self.covmatrix[np.isnan(self.covmatrix)]=0
#        self.covmatrix[np.isnan(self.covmatrix)]=9
        np.fill_diagonal(self.covmatrix,np.nan)

        self.entropies=entropies

    def transform(self,datain):
        if self.vic_type==self.VIC_TYPE_CIVIC:
            return self.transform_CIVIC(datain,False)

        if self.vic_type==self.VIC_TYPE_CIVIC_LINEAR:
            return self.transform_CIVIC_LINEAR(datain,False)

    def transform_CIVIC_LINEAR(self,datain,reverse):
        """
        datain: a numpy array of shape n_samples x n_features
        """
        datain=datain.reshape((datain.shape[0],-1))
        covmatrix=np.array(self.covmatrix,dtype=self.covmatrix.dtype)

        n_features=datain.shape[1]
        img_size=int(np.ceil(np.sqrt(n_features)))
        if img_size%2!=0: img_size+=1

        dataout=np.zeros((datain.shape[0],img_size,img_size))

        cx=cy=0
        if not reverse:
            dx=np.unravel_index(np.nanargmax(covmatrix),covmatrix.shape)[0]
        else:
            dx=np.unravel_index(np.nanargmin(covmatrix),covmatrix.shape)[0]

        dir=1
        dataout[:,cy,cx]=datain[:,dx]
        br=0
        while True:
            covmatrix[:,dx]=np.nan
            dxp=dx
            try:
                if not reverse:
                    dx=np.nanargmax(covmatrix[dx,:])
                else:
                    dx=np.nanargmin(covmatrix[dx,:])
            except:
                break
            if np.all(np.isnan(covmatrix[dx,:])): break
            covmatrix[dxp,:]=np.nan
            if not reverse:
                if np.nanmax(covmatrix[dx,:])<=0:
                    dx=np.unravel_index(np.nanargmax(covmatrix),covmatrix.shape)[0]
            else:
                if np.nanmin(covmatrix[dx,:])>=0:
                    dx=np.unravel_index(np.nanargmin(covmatrix),covmatrix.shape)[0]

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
            
    def transform_CIVIC(self,datain,reverse):
        """
        datain: a numpy array of shape n_samples x n_features
        """
        datain=datain.reshape((datain.shape[0],-1))
        covmatrix=np.array(self.covmatrix,dtype=self.covmatrix.dtype)

        n_features=datain.shape[1]
        img_size=int(np.ceil(np.sqrt(n_features)))
        if img_size%2!=0: img_size+=1

        dataout=np.zeros((datain.shape[0],img_size,img_size))

        cx=cy=img_size//2-1
        if not reverse:
            dx=np.unravel_index(np.nanargmax(covmatrix),covmatrix.shape)[0]
        else:
            dx=np.unravel_index(np.nanargmin(covmatrix),covmatrix.shape)[0]
        dir=4
        lem=1
        lemi=1
        dataout[:,cy,cx]=datain[:,dx]
        num_done=0
        while num_done<img_size*img_size:
            covmatrix[:,dx]=np.nan
            dxp=dx
            try:
                if not reverse:
                    dx=np.nanargmax(covmatrix[dx,:])
                else:
                    dx=np.nanargmin(covmatrix[dx,:])
            except: 
                break
            if np.all(np.isnan(covmatrix[dx,:])): break
            covmatrix[dxp,:]=np.nan
            if not reverse:
                if np.nanmax(covmatrix[dx,:])<=0:
                    dx=np.unravel_index(np.nanargmax(covmatrix),covmatrix.shape)[0]
            else:
                if np.nanmin(covmatrix[dx,:])>=0:
                    dx=np.unravel_index(np.nanargmin(covmatrix),covmatrix.shape)[0]

            if lemi>=lem-2 and dir==4:
                cy-=1
                cx-=1
                if lem>1: cx-=1
                lem+=2
                if lem>999: break
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
                        pass

            lemi+=1
            num_done+=1
            dataout[:,cy,cx]=datain[:,dx]

        return dataout
