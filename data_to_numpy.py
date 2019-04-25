import numpy as np
import os

data_path='Raw_data/train'

# Acc data files
accx_path = data_path+'/Acc_x.txt'
accy_path = data_path+'/Acc_y.txt'
accz_path = data_path+'/Acc_z.txt'

# Gyro data files
gyrox_path = data_path+'/Gyr_x.txt'
gyroy_path = data_path+'/Gyr_y.txt'
gyroz_path = data_path+'/Gyr_z.txt'

# Mag data files
magx_path = data_path+'/Mag_x.txt'
magy_path = data_path+'/Mag_y.txt'
magz_path = data_path+'/Mag_z.txt'

# Linear Acc data files
laccx_path = data_path+'/LAcc_x.txt'
laccy_path = data_path+'/LAcc_y.txt'
laccz_path = data_path+'/LAcc_z.txt'

# Gravity data files
grax_path = data_path+'/Gyr_x.txt'
gray_path = data_path+'/Gyr_y.txt'
graz_path = data_path+'/Gyr_z.txt'

# Orientation data files
oriw_path = data_path+'/Ori_w.txt'
orix_path = data_path+'/Ori_x.txt'
oriy_path = data_path+'/Ori_y.txt'
oriz_path = data_path+'/Ori_z.txt'

# Pressure data file
press_path= data_path+'/Pressure.txt'

# Labels
label_path = data_path+'/Label.txt'

# Training dataframe order
order_path = data_path+'/train_order.txt'

# Loading Files
print('Loading acc')
accx =  np.loadtxt(accx_path)
accy =  np.loadtxt(accy_path)
accz =  np.loadtxt(accz_path)
print('Loading gyro')
gyrox =  np.loadtxt(gyrox_path)
gyroy =  np.loadtxt(gyroy_path)
gyroz =  np.loadtxt(gyroz_path)
print('Loading mag')
magx =  np.loadtxt(magx_path)
magy =  np.loadtxt(magy_path)
magz =  np.loadtxt(magz_path)
print('Loading l_acc')
laccx =  np.loadtxt(laccx_path)
laccy =  np.loadtxt(laccy_path)
laccz =  np.loadtxt(laccz_path)
print('Loading grav')
grax =  np.loadtxt(grax_path)
gray =  np.loadtxt(gray_path)
graz =  np.loadtxt(graz_path)
print('Loading ori')
oriw =  np.loadtxt(oriw_path)
orix =  np.loadtxt(orix_path)
oriy =  np.loadtxt(oriy_path)
oriz =  np.loadtxt(oriz_path)
print('Loading pres')
press =  np.loadtxt(press_path)
print('Loading labels')
label= np.loadtxt(label_path)
print('Loading order')
order = np.loadtxt(order_path)


# Reordering
ordering = np.argsort(order)
accx = accx[ordering]
accy = accy[ordering]
accz = accz[ordering]

gyrox = gyrox[ordering]
gyroy = gyroy[ordering]
gyroz = gyroz[ordering]

magx = magx[ordering]
magy = magy[ordering]
magz = magz[ordering]

laccx = laccx[ordering]
laccy = laccy[ordering]
laccz = laccz[ordering]

grax = grax[ordering]
gray = gray[ordering]
graz = graz[ordering]

oriw = oriw[ordering]
orix = orix[ordering]
oriy = oriy[ordering]
oriz = oriz[ordering]

press = press[ordering]

label = label[ordering]


save_path = 'Extracted_data'
np.savez(save_path+'/Train_data', accx,accy,accz,gyrox,gyroy,gyroz,magx,magy,magz,laccx,laccy,laccz,grax,gray,graz,
        oriw,orix,oriy,oriz,press)
np.save(save_path+'/Train_labels', label)