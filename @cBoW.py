# source /var/python3envs/pytorch-0.3/bin/activate.csh

import math
import datetime
import torch
import numpy
import bz2
import sys
import re
import os
import torch.utils.data
from optparse import OptionParser
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

parser=OptionParser()
parser.add_option("-t","--train",dest="trainfile",
                  help="train ssv file",metavar="<training_file>")
parser.add_option("-v","--vectors",dest="vectfile",
                  help="vectors ssv file",metavar="<vectors_file>")
parser.add_option("-r","--test",dest="testfile",
                  help="test ssv file",metavar="<test_file>")
(options,args)=parser.parse_args()

fword=31
lword=71
hidden=256
batch=256
ilr=1e-3
dp=0.2
min_epochs=10
max_epochs=300
sample=0.1
gpuN=2

class DeepNet(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(DeepNet,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.feedf = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dp),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size,num_classes))
    def forward(self,x):
        out=self.feedf(x)
        return out

def to_var(x):
    if torch.cuda.is_available():
        x=x.cuda()
    return Variable(x)

def setParams(x):
    instances=len(x[:,0])
    val_size=int(instances*sample)
    return instances,val_size

def to_avg(x):
    y = torch.zeros(len(x),vsize).type(torch.FloatTensor)
    for b in range(len(x)):
        z = 0
        for s in range(len(x[0])):
            if x[b][s] != 0:
                y[b] = y[b] + w2vTensor[x[b][s]]
                z += 1
        if z != 0:        
            y[b] = y[b]/z
    return y

if torch.cuda.is_available():
    torch.cuda.set_device(gpuN)

if options.vectfile is not None:
    if options.trainfile is not None:
        print()
        print(" GPU available:",torch.cuda.is_available())
        now=datetime.datetime.now()
        sys.stdout.write(" loading vectors: ")
        sys.stdout.flush()
    if re.search(".bz2",options.vectfile) is not None:
        with bz2.open(options.vectfile,'rt',encoding='utf-8') as f:
            vsize=f.readline().rstrip().split(" ")
    else:
        with open(options.vectfile,'r',encoding='utf-8') as f:
            vsize=f.readline().rstrip().split(" ")
    vsize=int(len(vsize)-1)
    smooth=numpy.zeros(vsize,dtype=float)
    w2vTensor=numpy.append([smooth],numpy.loadtxt(options.vectfile,skiprows=0,usecols=range(1,vsize+1),dtype=float,ndmin=2),axis=0)
    w2vTensor=torch.from_numpy(w2vTensor).type(torch.FloatTensor)
    if options.trainfile is not None:
        prev=now
        now=datetime.datetime.now()
        print(round((now-prev).total_seconds()/60,1),"min")
else:
    print()
    print(" include w2v file.")

if options.trainfile is not None:

    plague=numpy.loadtxt(options.trainfile,comments="######",dtype=str,ndmin=2)
    print(" a plague of",len(plague),"m0ths")
    print()
    nowAll=datetime.datetime.now()
    for m in plague:
        ssv=m[0]
        m0th=m[1]

        if re.search(".bz2",ssv) is not None:
            with bz2.open(ssv,'rt',encoding='utf-8') as f:
                prior=list(map(float,f.readline().split(" ")))
        else:
            with open(ssv,'r',encoding='utf-8') as f:
                prior=list(map(float,f.readline().split(" ")))
        classes=int(len(prior))

        nowT=datetime.datetime.now()
        sys.stdout.write("  ["+os.path.basename(ssv)+"]")
        D=numpy.loadtxt(ssv,skiprows=1,usecols=range(fword,lword),dtype='int32',ndmin=2)
        G=numpy.loadtxt(ssv,skiprows=1,usecols=(0,),dtype='int32',ndmin=1)
        instances,val_size = setParams(D)

        #trainD,devD,trainG,devG=train_test_split(D,G,stratify=G,test_size=sample)
        trainD,devD,trainG,devG=train_test_split(D,G,test_size=sample)
        D=None
        G=None
        devDataTensor=to_avg(torch.from_numpy(devD).type(torch.IntTensor))
        devGoldTensor=torch.from_numpy(devG).type(torch.IntTensor)
        trainDataTensor=to_avg(torch.from_numpy(trainD).type(torch.IntTensor))
        trainGoldTensor=torch.from_numpy(trainG).type(torch.IntTensor)
        trainDataGoldTensor=torch.utils.data.TensorDataset(trainDataTensor,trainGoldTensor)
        devDataGoldTensor=torch.utils.data.TensorDataset(devDataTensor,devGoldTensor)
        devloader=torch.utils.data.DataLoader(devDataGoldTensor,batch_size=batch)
        trainD=None
        devD=None
        trainG=None
        devG=None
        devDataTensor=None
        devGoldTensor=None
        trainDataTensor=None
        trainGoldTensor=None

        prevT=nowT
        nowT=datetime.datetime.now()
        print(" loading: "+str(round((nowT-prevT).total_seconds(),1))+" sec")

        for r in [1,2,3]:
            nowT=datetime.datetime.now()
            NNet=DeepNet(vsize,hidden,classes)
            loss_fn=torch.nn.CrossEntropyLoss()
            optimizer=torch.optim.Adam(NNet.parameters(),lr=ilr)
            if torch.cuda.is_available():
                NNet=NNet.cuda()
            accD_best=0
            lossD_best=1000
            rounds=0
            for epoch in range(max_epochs):
                NNet.train()
                loader=torch.utils.data.DataLoader(trainDataGoldTensor,batch_size=batch,shuffle=True)
                lossD = 0
                T = 0
                for x,y in loader:
                    inp=to_var(x)
                    tar=to_var(y)
                    tar_pred=NNet(inp)
                    loss=loss_fn(tar_pred,tar-1)
                    lossD=lossD+loss.data[0]
                    T=T+1
                    NNet.zero_grad()
                    loss.backward()
                    optimizer.step()
                NNet.eval()
                lossD=round(lossD/T,6)
                correct=0
                all=0
                for x,y in devloader:
                    dinp=to_var(x)
                    dtar=to_var(y)
                    predTensorV=NNet(dinp)
                    row=0
                    for x in predTensorV:
                        maxA,a=torch.max(x,0)
                        b=dtar[row]
                        if (a.data[0]==b.data[0]-1):
                            correct+=1 
                        row+=1
                        all+=1
                accD=round(100*(float(correct)/(all)),2)
                if (accD_best<=accD and lossD_best>=lossD):
                    ep=epoch
                    torch.save(NNet.state_dict(),m0th+str(r))
                    rounds=0
                    accD_best=accD
                    lossD_best=lossD
                else:
                    rounds+=1
                if epoch+1>=min_epochs and rounds>=10:
                    break

            prevT=nowT
            nowT=datetime.datetime.now()
            print("     devAcc:",accD_best,"epochs:",ep+1,"instances:",instances,"training:",round((nowT-prevT).total_seconds(),1),"sec")

    prevAll=nowAll
    nowAll=datetime.datetime.now()
    print()
    print(" total time:",round((nowAll-prevAll).total_seconds()/60,1),"min")
    print(" time per mention:",round((nowAll-prevAll).total_seconds()/len(plague),1),"sec")
    print()

if options.testfile is not None:

    plague=numpy.loadtxt(options.testfile,comments="######",dtype=str,ndmin=2)
    for m in plague:
        ssv=m[0]
        m0th=m[1]
        if re.search(".bz2",ssv) is not None:
            with bz2.open(ssv,'rt',encoding='utf-8') as f:
                prior=list(map(float,f.readline().split(" ")))
        else:
            with open(ssv,'r',encoding='utf-8') as f:
                prior=list(map(float,f.readline().split(" ")))
        classes=int(len(prior))
        data=numpy.loadtxt(ssv,skiprows=1,usecols=range(fword,lword),dtype=int,ndmin=2)
        ids=numpy.loadtxt(ssv,skiprows=1,usecols=(0,),dtype=str,ndmin=1)
        inputs=to_var(to_avg(torch.from_numpy(data).type(torch.IntTensor)))
        data=None

        NNet=DeepNet(vsize,hidden,classes)
        NNet.load_state_dict(torch.load(m0th))
        NNet.eval()
        if torch.cuda.is_available():
            NNet=NNet.cuda()

        lsfm=torch.nn.LogSoftmax(dim=1)
        targets_pred=lsfm(NNet(inputs))
        for k,i in enumerate(ids):
            sys.stdout.write(i)
            sys.stdout.flush()
            for val in targets_pred.data[k]:
                sys.stdout.write(" ")
                sys.stdout.flush()
                sys.stdout.write(str(val))
                sys.stdout.flush()
            print()
    
