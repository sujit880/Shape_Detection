#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def sort_items(a):
    path="./images5"
    return int(a[len(path)+1:len(a)-4])


# In[3]:


path="./images5"
images = glob.glob(path+"/*.png")
images.sort(key=sort_items)
#a=sorted(images, key=sort_items)
#print(images)


# In[4]:


def thrsh1(img,t):
    inpt=img.copy()
    for i in range (len(inpt)):
        for j in range (len(inpt[i])):
            if inpt[i][j]>t:
                inpt[i][j]=t
    ret, thresh1 = cv2.threshold(inpt, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("thresh1"+str(int(t))+".png",thresh1)
    plt.imshow(thresh1)
    return ret

def thrsh2(img,t):
    inpt=img.copy()
    for i in range(len(inpt)):
        for j in range (len(inpt[i])):
            if inpt[i][j]<t:
                inpt[i][j]=t
    ret, thresh1 = cv2.threshold(inpt, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("thresh2"+str(int(t))+".png",thresh1)
    plt.imshow(thresh1)
    return ret,thresh1

def setpx(img,t1,t2,t3):
    inpt=img.copy()
    inpt1 = img.copy()
    inpt2 = img.copy()
    inpt3 = img.copy()
    for i in range(len(inpt)):
        for j in range (len(inpt[i])):
            if (inpt[i][j]<t1):
                inpt[i][j]=0
            elif(inpt[i][j]<t2):
                inpt[i][j]=t1
            elif (inpt[i][j]<t3):
                inpt[i][j]=t2
            else:
                inpt[i][j]=t3
    cv2.imwrite("Final.png",inpt)
    plt.imshow(inpt)
    return inpt

def writepx(img,t1,t2,t3):
    inpt=img.copy()
    inpt1 = img.copy()
    inpt2 = img.copy()
    inpt3 = img.copy()
    for i in range(len(inpt)):
        for j in range (len(inpt[i])):
            if (inpt1[i][j]==t1):
                inpt1[i][j]=t1
            else:
                inpt1[i][j]=0
            if(inpt2[i][j]==t2):
                inpt2[i][j]=t2
            else:
                inpt2[i][j]=0
            if (inpt3[i][j]==t3):
                inpt3[i][j]=t3
            else:
                inpt3[i][j]=0
                
    cv2.imwrite("inpt1.png",inpt1)
    cv2.imwrite("inpt2.png",inpt2)
    cv2.imwrite("inpt3.png",inpt3)


# In[5]:


def im_show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 


# In[10]:


def crop(img):
    image = cv2.imread(img)
    #imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    #pts = np.array([[1300,580], [1700,580], [1600,720], [1400,720]])
    pts = np.array([[1350,590], [1700,590], [1700,720], [1350,720]])
    color = 255
    cv2.fillPoly(mask, [pts], color)
    masked = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = image[y:y+h, x:x+w].copy()
    print(croped.shape, image.shape)
    return croped, image


# In[11]:


x,y = 1000,10
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2
color=(0,0,0)
count=0
out_path="./out5/"
for img in images:
    #path="./images5"
    print(img[len(path)+1:])
    croped, original = crop(img)
    x1=x+croped.shape[1]+200
    y1= y +20
    red_box = cv2.rectangle(original, (x1, y1), (x1+300, y1+100), (255, 255, 255), -1)
    original[y:y+croped.shape[0],x:x+croped.shape[1]] = croped
    org=(x1+5,y1+60)
    im_show(croped)
    inpt=input("enter count value")
    if(inpt==''):
        inpt=input("enter correct count value")
    count += int(inpt)
    image = cv2.putText(original, str(count), org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(out_path+img[len(path)+1:],image)
    #im_show(original)
    #break;


# In[ ]:




