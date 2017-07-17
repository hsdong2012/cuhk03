# -*- coding: utf-8 -*-
# --------------------------------------------------------
#
# --------------------------------------------------------

"""
A script to image dataset from original CUHK03 mat file.
"""

import h5py
import numpy as np
from PIL import Image
import scipy.misc
import scipy.io as sio
import argparse
import sys
import itertools as it
import os

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Script to Create HDF5 dataset.')
    parser.add_argument('--mat',
                        dest='mat_file_path',
                        help='Original CUHK03 file path.',
                        required=True)
    args = parser.parse_args()

    return args

def create_dataset(file_path):

    with h5py.File(file_path,'r') as f:

        #use camera pair 1-5
        for i in xrange(5):
            for k in xrange(f[f['labeled'][0][i]][0].size):
                print i,k
                train_file_path = 'train/id'+str(k+i*1000)+'/'
                train_directory = os.path.dirname(train_file_path)
                if not os.path.exists(train_directory):
                    os.makedirs(train_directory)

                val_file_path = 'val/id'+str(k+i*1000)+'/'
                val_directory = os.path.dirname(val_file_path)
                if not os.path.exists(val_directory):
                    os.makedirs(val_directory)

                for j in it.chain(xrange(1, 5), xrange(6, 10)):
                    if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                        img1 = np.array(f[f[f['labeled'][0][i]][j][k]][:]).transpose(2,1,0)
                        img1 = scipy.misc.imresize(img1, (224,224))
                        scipy.misc.imsave(train_directory+'/'+str(j)+'.png', img1)

                    if len(f[f[f['detected'][0][i]][j][k]].shape) == 3:
                        img2 = np.array(f[f[f['detected'][0][i]][j][k]][:]).transpose(2,1,0)
                        img2 = scipy.misc.imresize(img2, (224,224))
                        scipy.misc.imsave(train_directory+'/'+str(j+10)+'.png', img2)

                for j in it.chain(xrange(1), xrange(5, 6)):
                    if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                        img3 = np.array(f[f[f['labeled'][0][i]][j][k]][:]).transpose(2,1,0)
                        img3 = scipy.misc.imresize(img3, (224,224))
                        scipy.misc.imsave(val_directory+'/'+str(j)+'.png', img3)
                    if len(f[f[f['detected'][0][i]][j][k]].shape) == 3:
                        img4 = np.array(f[f[f['detected'][0][i]][j][k]][:]).transpose(2,1,0)
                        img4 = scipy.misc.imresize(img4, (224,224))
                        scipy.misc.imsave(val_directory+'/'+str(j+10)+'.png', img4)


if __name__ == '__main__':

    args = parse_args()
    create_dataset(args.mat_file_path)
