#coding:utf-8
import sys
import os
reload(sys)
sys.setdefaultencoding( "utf-8" )
import PIL.Image as image
import numpy as np
import random


def storeAllPicInMe(flagOfO,flagOfB,size):
    path1="/home/alex/PycharmProjects/oracleCNN/repository/size"+str(size)+"x"+str(size)+"/bronze/"
    path2="/home/alex/PycharmProjects/oracleCNN/repository/size"+str(size)+"x"+str(size)+"/oracle/"
    X=list()
    Y=list()

    numOferror=0
    parents =os.listdir(path1)
    if(flagOfB):
        for pic in parents:
            picPath=os.path.join(path1,pic)

            try:
                im = image.open(picPath)
                array = np.array(im)

                ##im.show()
                array = np.array(im)/128.0-1
                Y.append(parseName(pic))
                X.append(array)


            except IOError:
                numOferror=numOferror+1
                print 'error '+str(pic)
    parents =os.listdir(path2)
    if(flagOfO):
        for pic in parents:
            picPath=os.path.join(path2,pic)

            try:
                im = image.open(picPath)

                ##im.show()
                array = np.array(im)/128.0-1
                Y.append(parseName(pic))
                X.append(array)

            except IOError:
                numOferror=numOferror+1
                print 'error '+str(pic)


    return(X,Y)

def parseName(name):
    index=int(name[-8:-4],16)
    return index

def convertIndexTovector(l,Y):
    newY=list()
    for i in range(len(Y)):
        newTarget=np.zeros((l),int)
        newTarget[Y[i]]=1
        newY.append(newTarget)

    return newY

def convertDiscretToContinueT(target):
    dictory=list()
    for i in range(len(target)):
        nofound=True
        for j in range(len(dictory)):
            if dictory[j]==target[i]:
                target[i]=j
                nofound=False
        if nofound:
            dictory.append(target[i])
            target[i]=len(dictory)-1
    return dictory,target

def shuffle(X,Y):
    combine=list()
    for i in range(len(X)):
        combine.append((X[i],Y[i]))
    random.shuffle(combine)

    newX=list()
    newY=list()
    for i in range(len(combine)):
        newX.append(combine[i][0])
        newY.append(combine[i][1])

    return newX,newY




def split2testAndTrain(X,Y,batch_size):
    print len(X[0])
    numForTrain=len(X)/batch_size/10*128*9

    train_data_x=X[0:numForTrain]
    train_data_y=Y[0:numForTrain]
    test_data_x=X[numForTrain:]
    test_data_y=Y[numForTrain:]

    return(train_data_x,train_data_y,test_data_x,test_data_y)



def statistic():



    pathb="/home/alex/PycharmProjects/oracleCNN/repository/size28x28/bronze/"
    patho="/home/alex/PycharmProjects/oracleCNN/repository/size28x28/oracle/"
    pathl="/home/alex/PycharmProjects/oracleCNN/repository/size28x28/lst/"
    pathse="/home/alex/PycharmProjects/oracleCNN/repository/size28x28/seal/"
    namesForb=list()
    namesForo=list()
    namesForl=list()
    namesFors=list()
    numForb=list()
    numForo=list()
    numForl=list()
    numFors=list()
    paths=[pathb,patho,pathl,pathse]
    names=[namesForb,namesForo,namesForl,namesFors]
    nums=[numForb,numForo,numForl,numFors]
    for i in range(len(paths)):
        parents = os.listdir(paths[i])
        for pic in parents:
            name=pic[-8:-4]
            noFound = True
            for j in range(len(names[i])):
                if name==names[i][j]:
                    nums[i][j]=nums[i][j]+1
                    noFound=False
                    break
            if noFound:
                names[i].append(name)
                nums[i].append(1)
    nameForall=list()
    numForall=list()


    for i in range(4):
        for j in range(len(names[i])):
            noFound = True
            for k in range(len(nameForall)):
                if names[i][j]==nameForall[k]:
                    numForall[k]=numForall[k]+1
                    noFound=False
                    break
            if noFound:
                nameForall.append(names[i][j])
                numForall.append(1)

    stat=np.zeros((4,len(nameForall)),np.int)

    for i in range(4):
        for j in range(len(names[i])):
            name=names[i][j]
            num=nums[i][j]
            for k in range(len(nameForall)):
                if name==nameForall[k]:
                    stat[i][k]=num
                    break




    file=open('../repository/statistic','w')
    for i in range(len(nameForall)):
        line=nameForall[i]+','+str(stat[0][i])+","+str(stat[1][i])+","+str(stat[2][i])+","+str(stat[3][i])+"\n"
        file.write(line)
    file.close()





    print("total num of character"+str(len(nameForall)))
    print("num of oracle character"+str(len(names[0])))
    print("num of bronze character"+str(len(names[1])))
    print("num of lst character"+str(len(names[2])))
    print("num of seal character"+str(len(names[3])))


def readStat():
    name=list()
    num=list()
    for line in open('../repository/statistic','r'):
        items=line.split(',')
        name.append(items[0])
        numtemp=list()
        for i in range(len(items)-1):
            numtemp.append(int(items[i+1]))
        num.append(numtemp)

    return (name,num)


##print parseName("J00001_4e00.jpg")
##(X,Y,dictory)=storeAllPicInMe()
'''
X=[1,2,3,4]
Y=[2,4,6,8]
print " cd"
(X,Y)=shuffle(X,Y)
print X
print Y
print "done"
'''
##print len(X)
##print len(dictory)
##statistic()
##print readStat()[1]

##print [1,2,34][1:]

'''
(X,Y,dictory)=storeAllPicInMe()
X,Y=shuffle(X,Y)
print Y[0]
print Y[1]

'''
