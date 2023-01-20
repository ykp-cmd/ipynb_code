import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#rotate the image
def imageRotate(img):
  h, w = img.shape[:2]
  size = (w, h)

  #set rotation angle
  angle = 10
  angle_rad = angle/180.0*np.pi

  #calculate image size after rotation
  w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
  h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
  size_rot = (w_rot, h_rot)

  #rotate around the centre of the original image
  center = (w/2, h/2)
  scale = 1.0
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

  #add a parallel shift(rotation + translation)
  affine_matrix = rotation_matrix.copy()
  affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
  affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2

  #return rotated image
  img = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
  return img
  
#create mask
def createMask(img):
  #convert to HSV
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  #binarise
  bin_img = cv2.inRange(hsv, (28, 78, 40), (230, 230, 230))

  #extract contours
  contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  #get the contour with the largest area.
  contour = max(contours, key=lambda x: cv2.contourArea(x))
  
  #create mask image
  mask = np.zeros_like(bin_img)
  maskImg =  cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
  return maskImg
  
#read facade image
fgImgs = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/CycleGAN_Project/pytorch-CycleGAN-and-pix2pix/datasets/capycopied/fgcapy/*.jpg'), key=os.path.getsize)
#read background image
bgImgs = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/CycleGAN_Project/pytorch-CycleGAN-and-pix2pix/datasets/capycopied/bgcapy/*.jpg'), key=os.path.getsize)

fgImg = random.choice(fgImgs) #pick an image up randomly
loadedFgImg = cv2.imread(fgImg)
loadedFgImg = imageRotate(loadedFgImg)
loadedFgImg = cv2.cvtColor(loadedFgImg, cv2.COLOR_BGR2RGB) #convert BGR into RGB

#resize image at random
size = random.choice(range(1000, 3000))
rs1 = cv2.resize(loadedFgImg, (size, size))

mask = createMask(rs1)

#process in the same way as fgImg
bgImg = random.choice(bgImgs) 
loadedBgImg = cv2.imread(bgImg)
loadedBgImg = cv2.cvtColor(loadedBgImg, cv2.COLOR_BGR2RGB)
rs2 = cv2.resize(loadedBgImg, (4000, 4000))

#set the position for pasting
x = random.choice(range(1000, 2500))
y = random.choice(range(1000, 3000))

#width and height take the common part of the foreground and background images
w = min(rs1.shape[1], rs2.shape[1] - x)
h = min(rs1.shape[0], rs2.shape[0] - y)

#an aria where you'd synthesize
fg_roi = rs1[:h, :w]
bg_roi = rs2[y : y + h, x : x + w]

#synthesise what is produced in the last two processes.
bg_roi[:] = np.where(mask[:h, :w, np.newaxis] == 0, bg_roi, fg_roi)
plt.imshow(rs2)
plt.show()

#fname = os.path.splitext(os.path.basename(fgImg))[0] + "+" + os.path.splitext(os.path.basename(bgImg))[0]
#rs2 = cv2.cvtColor(rs2, cv2.COLOR_BGR2RGB)
#cv2.imwrite(f'/content/drive/MyDrive/Colab Notebooks/CycleGAN_Project/pytorch-CycleGAN-and-pix2pix/results/opencv/{fname}.jpg', rs2) #save converted file
  
