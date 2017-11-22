#coding:utf-8
import sys
import os
reload(sys)
sys.setdefaultencoding( "utf-8" )
import PIL.Image as image
import numpy as np






## 40x60
def allGifAndResize():
    numOfWrongPic=0
    totallPic=0

    path="/home/alex/PycharmProjects/oracleCNN/repository/bronzePic"
    savepath="/home/alex/PycharmProjects/oracleCNN/repository/size64x64/bronze/"
    parents =os.listdir(path)
    for pic in parents:
        path2save=savepath+pic
        picPath=os.path.join(path,pic)
        totallPic=totallPic+1
        try:
            im=image.open(picPath).convert('L')
            im = im.resize((64, 64))
            im.save(path2save, "jpeg")

        except IOError:
            numOfWrongPic=numOfWrongPic+1



    print totallPic
    print numOfWrongPic


'''
  	

Convert between PIL image and NumPy ndarray

image = Image.open(“ponzo.jpg”)   # image is a PIL image 
array = numpy.array(image)  

'''
print "hello world"
allGifAndResize()