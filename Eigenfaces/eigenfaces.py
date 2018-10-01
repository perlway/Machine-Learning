# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 22:18:28 2018

@author: vk186043
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


np.seterr(divide='ignore', invalid='ignore')
images_path=r"./dataset/"

images_matrix = np.empty(shape=(65536,520))
images_matrix_centred = np.empty(shape=(65536,520))
'''
Read images in the images_path directory
into an array in gray mode
'''  
column = 0
for file in os.listdir(images_path):
    im = Image.open(os.path.join(images_path, file))
    im = im.convert("L")
    img_col = np.array(im).flatten()
    images_matrix[:,column] = img_col[:]
    column = column+1

'''
Mean image of the input images
'''
mean_img_col = np.sum(images_matrix, axis=1) / 520
plt.imshow(np.reshape(mean_img_col,[256,256]), cmap = plt.get_cmap("gray"))
plt.figure()


for j in range(0, 519):
    images_matrix_centred[:,j] = images_matrix[:,j] - mean_img_col[:]
    
'''
Eigen vectors for the images 
'''
 
covariance_matrix = np.matrix(images_matrix_centred.transpose()) * np.matrix(images_matrix_centred)
covariance_matrix = covariance_matrix/520
evalues, evectors = np.linalg.eig(covariance_matrix)
evectors = evectors.transpose()
sortindex = evalues.argsort()[::-1]
evectors = evectors[sortindex]
evalues  = evalues[sortindex]

'''
Plot of the significant eigen faces , first 9 eigen faces
'''

evectors_img = images_matrix_centred * evectors
fig = plt.figure()
fig.set_size_inches(10, 10,forward=True)
ax1 = fig.add_subplot(3,3,1)
ax1.imshow(np.reshape(evectors_img[:,0],[256,256]), cmap = plt.get_cmap("gray"))
ax2 = fig.add_subplot(3,3,2)
ax2.imshow(np.reshape(evectors_img[:,1],[256,256]), cmap = plt.get_cmap("gray"))
ax3 = fig.add_subplot(3,3,3)
ax3.imshow(np.reshape(evectors_img[:,2],[256,256]), cmap = plt.get_cmap("gray"))
ax4 = fig.add_subplot(3,3,4)
ax4.imshow(np.reshape(evectors_img[:,3],[256,256]), cmap = plt.get_cmap("gray"))
ax5 = fig.add_subplot(3,3,5)
ax5.imshow(np.reshape(evectors_img[:,4],[256,256]), cmap = plt.get_cmap("gray"))
ax6 = fig.add_subplot(3,3,6)
ax6.imshow(np.reshape(evectors_img[:,5],[256,256]), cmap = plt.get_cmap("gray"))
ax7 = fig.add_subplot(3,3,7)
ax7.imshow(np.reshape(evectors_img[:,6],[256,256]), cmap = plt.get_cmap("gray"))
ax8 = fig.add_subplot(3,3,8)
ax8.imshow(np.reshape(evectors_img[:,7],[256,256]), cmap = plt.get_cmap("gray"))
ax9 = fig.add_subplot(3,3,9)
ax9.imshow(np.reshape(evectors_img[:,8],[256,256]), cmap = plt.get_cmap("gray"))
plt.show()



count = np.arange(1,519,10);
length = len(count)
MSE = np.empty(shape=(length,1))
mse_counter = 0;
'''
Loop with eigen faces counter to calculate the MSE
'''
for i in count:
    evalues  = evalues[0:i]
    evectors_temp = evectors[0:i].transpose()
    evectors_img = images_matrix_centred * evectors_temp
    norms = np.linalg.norm(evectors_img, axis=0)
    evectors_img = evectors_img/norms
    evectors_img = evectors_img.transpose()
    Weights =  evectors_img * images_matrix_centred
    images = (Weights.transpose() * evectors_img )
    images = images.transpose()
    diff = images - images_matrix
    MSE[mse_counter] = np.sum(np.square(diff))/65536
    mse_counter += 1

plt.plot(count,MSE)
plt.xlabel('Eigen Faces')
plt.ylabel('Mean Square Error')
plt.show()


'''
Reconstructed image with all the eigen faces
'''



plt.imshow(np.reshape(images_matrix[:,0],[256,256]), cmap = plt.get_cmap("gray"))
plt.figure()
plt.imshow(np.reshape(images[:,0],[256,256]), cmap = plt.get_cmap("gray"))
plt.figure()

    
    
    
    
