#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy as np
import matplotlib.pyplot as plt
import random as rd

dataspiral=pandas.read_csv('spiral.csv',header=None)
X = dataspiral.iloc[:, [0, 1]].values


# In[3]:


a=X.shape[0] 
b=X.shape[1] 
K=3
Centroids=np.array([]).reshape(b,0) 
for i in range(K):
    acak=rd.randint(0,a-1)
    Centroids=np.c_[Centroids,X[acak]]
plt.scatter(dataspiral[0],dataspiral[1],c='blue') 
plt.scatter(Centroids[0],Centroids[1],c='red')
plt.show()


# In[4]:


for i in range(1000):
    i=i+1
    centroid1=[]
    centroid2=[]
    centroid3=[]
    xdata=[]
    ydata=[]
    hasil1fix=[]
    hasil2fix=[]
    hasil3fix=[]
    korcen1x=[]
    korcen2x=[]
    korcen3x=[]
    korcen1y=[]
    korcen2y=[]
    korcen3y=[]
    for i in range (len(Centroids)):
        centroid1.append(Centroids[i][0])
    for i in range (len(Centroids)):
        centroid2.append(Centroids[i][1])
    for i in range (len(Centroids)):
        centroid3.append(Centroids[i][2])
    for i in range (len(X)):
        xdata.append(X[i][0])
    for i in range (len(X)):
        ydata.append(X[i][1])
    j=0
    for j in range(311):
        j=j+1
        x=(xdata[j]-centroid1[0])**2
        y=(ydata[j]-centroid1[1])**2
        hasil1=(np.sqrt(x+y))
        hasil1fix.append(hasil1)

        x1=(xdata[j]-centroid2[0])**2
        y1=(ydata[j]-centroid2[1])**2
        hasil2=(np.sqrt(x1+y1))
        hasil2fix.append(hasil2)

        x2=(xdata[j]-centroid3[0])**2
        y2=(ydata[j]-centroid3[1])**2
        hasil3=(np.sqrt(x2+y2))
        hasil3fix.append(hasil3)

    for j in range(310):
        j=j+1
        if hasil1fix[j]>hasil3fix[j]>hasil2fix[j]:
            korcen2x.append(xdata[j])
            korcen2y.append(ydata[j])
        if hasil3fix[j]>hasil1fix[j]>hasil2fix[j]:
            korcen2x.append(xdata[j])
            korcen2y.append(ydata[j])

        if hasil1fix[j]>hasil2fix[j]>hasil3fix[j]:
            korcen3x.append(xdata[j])
            korcen3y.append(ydata[j])
        if hasil2fix[j]>hasil1fix[j]>hasil3fix[j]:
            korcen3x.append(xdata[j])
            korcen3y.append(ydata[j])

        if hasil3fix[j]>hasil2fix[j]>hasil1fix[j]:
            korcen1x.append(xdata[j])
            korcen1y.append(ydata[j])
        if hasil2fix[j]>hasil3fix[j]>hasil1fix[j]:
            korcen3x.append(xdata[j])
            korcen3y.append(ydata[j])
    w=np.mean(korcen1x)
    s=np.mean(korcen1y)
    u=np.mean(korcen2x)
    e=np.mean(korcen2y)
    d=np.mean(korcen3x)
    c=np.mean(korcen3y)
    Centroids[0][0]=w
    Centroids[0][1]=u
    Centroids[0][2]=d
    Centroids[1][0]=s
    Centroids[1][1]=e
    Centroids[1][2]=c
    plt.scatter(korcen1x,korcen1y,c='blue') 
    plt.scatter(korcen2x,korcen2y,c='red')
    plt.scatter(korcen3x,korcen3y,c='green')
    plt.scatter(Centroids[0],Centroids[1],c='black')
    plt.show()


# In[5]:


sum1=np.sum(korcen1x)
sum2=np.sum(korcen1y)
sum3=np.sum(korcen2x)
sum4=np.sum(korcen2y)
sum5=np.sum(korcen3x)
sum6=np.sum(korcen3y)


# In[6]:


korcen1x.append(Centroids[0][0])
korcen1y.append(Centroids[1][0])
korcen2x.append(Centroids[0][0])
korcen2y.append(Centroids[1][0])
korcen3x.append(Centroids[0][0])
korcen3y.append(Centroids[1][0])
sum1=np.sum(korcen1x)
sum2=np.sum(korcen1y)
sum3=np.sum(korcen2x)
sum4=np.sum(korcen2y)
sum5=np.sum(korcen3x)
sum6=np.sum(korcen3y)
kumpulansum=[]
kumpulansum.append(sum1)
kumpulansum.append(sum2)
kumpulansum.append(sum3)
kumpulansum.append(sum4)
kumpulansum.append(sum5)
kumpulansum.append(sum6)
print("SSE= ",kumpulansum)


# In[7]:


import csv
kump1=[]
kump2=[]
kump3=[]
X1 = dataspiral.iloc[:, [0, 1, 2]].values
for i in range(len(X1)):
    if X1[i][2]==3:
        kump3.append(X1[i])
    if X1[i][2]==2:
        kump2.append(X1[i])
    if X1[i][2]==1:
        kump1.append(X1[i])


# In[8]:


titikrandom=rd.randint(0,100)
for i in range(len(kump3)):
    plt.scatter(kump3[i][0],kump3[i][1],c='blue')
for i in range(len(kump2)):
    plt.scatter(kump2[i][0],kump2[i][1],c='red') 
for i in range(len(kump1)):
    plt.scatter(kump1[i][0],kump1[i][1],c='green') 
plt.scatter(kump3[titikrandom][0],kump3[titikrandom][1],c='yellow')
plt.scatter(kump2[titikrandom][0],kump2[titikrandom][1],c='yellow')
plt.scatter(kump1[titikrandom][0],kump1[titikrandom][1],c='yellow')
plt.show()


# In[9]:


centroid2x=[]
centroid2y=[]
centroid2x.append(kump1[titikrandom][0])
centroid2y.append(kump1[titikrandom][1])
centroid2x.append(kump2[titikrandom][0])
centroid2y.append(kump2[titikrandom][1])
centroid2x.append(kump3[titikrandom][0])
centroid2y.append(kump3[titikrandom][1])


# In[10]:


for i in range(10):
    i=i+1
    listkump1x=[]
    listkump1y=[]
    listkump2x=[]
    listkump2y=[]
    listkump3x=[]
    listkump3y=[]
    for i in range(len(kump1)):
        listkump1x.append(kump1[i][0])
        listkump1x.append(centroid2x[0])
        listkump1y.append(kump1[i][1])
        listkump1y.append(centroid2y[0])
    for i in range(len(kump2)):
        listkump2x.append(kump2[i][0])
        listkump2x.append(centroid2x[1])
        listkump2y.append(kump2[i][1])
        listkump2y.append(centroid2y[1])
    for i in range(len(kump3)):
        listkump3x.append(kump3[i][0])
        listkump3x.append(centroid2x[2])
        listkump3y.append(kump3[i][1])
        listkump3y.append(centroid2y[2])

    w1=np.mean(listkump1x)
    s2=np.mean(listkump1y)
    u3=np.mean(listkump2x)
    e4=np.mean(listkump2y)
    d5=np.mean(listkump3x)
    c6=np.mean(listkump3y)

    centroid2x[0]=w1
    centroid2y[0]=s2
    centroid2x[1]=u3
    centroid2y[1]=e4
    centroid2x[2]=d5
    centroid2y[2]=c6
    for i in range(len(kump3)):
        plt.scatter(kump3[i][0],kump3[i][1],c='blue')
    for i in range(len(kump2)):
        plt.scatter(kump2[i][0],kump2[i][1],c='red') 
    for i in range(len(kump1)):
        plt.scatter(kump1[i][0],kump1[i][1],c='green') 
    plt.scatter(centroid2x[0],centroid2y[0],c='yellow')
    plt.scatter(centroid2x[1],centroid2x[1],c='yellow')
    plt.scatter(centroid2x[2],centroid2y[2],c='yellow')
    plt.show()


# In[11]:


summ1=np.sum(listkump1x)
summ2=np.sum(listkump1y)
summ3=np.sum(listkump2x)
summ4=np.sum(listkump2y)
summ5=np.sum(listkump3x)
summ6=np.sum(listkump3y)
kumpulansum1=[]
kumpulansum1.append(summ1)
kumpulansum1.append(summ2)
kumpulansum1.append(summ3)
kumpulansum1.append(summ4)
kumpulansum1.append(summ5)
kumpulansum1.append(summ6)
print("SSE= ",kumpulansum1)


# In[ ]:




