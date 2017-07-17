# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Group a has train, test, and val datasets
# Group b has test and val datasets
# --------------------------------------------------------

"""
A script to create a HDF5 dataset from original CUHK03 mat file.
"""

import h5py
import numpy as np
from PIL import Image
import argparse
import sys
import itertools as it

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

    with h5py.File(file_path,'r') as f, h5py.File('cuhk-03.h5') as fw:

        val_index = (f[f['testsets'][0][0]][:].T - 1).tolist()
        tes_index = (f[f['testsets'][0][1]][:].T - 1).tolist()

        # group a from 'labeled'
        fwa = fw.create_group('a')
        fwat = fwa.create_group('train')
        fwav1 = fwa.create_group('val_1')
        fwav2 = fwa.create_group('val_2')
        fwat_id = fwa.create_group('train_id')
        fwav1_id = fwa.create_group('val_1_id')
        fwav2_id = fwa.create_group('val_2_id')
        #group b from 'detected'
        fwb = fw.create_group('b')
        fwbt = fwb.create_group('train')
        fwbv1 = fwb.create_group('val_1')
        fwbv2 = fwb.create_group('val_2')
        fwbt_id = fwb.create_group('train_id')
        fwbv1_id = fwb.create_group('val_1_id')
        fwbv2_id = fwb.create_group('val_2_id')

        a_temp = []
        a_temp_t_id = []
        a_temp_v1_id = []
        a_temp_v2_id = []
        a_count_t = 0
        a_count_v1 = 0
        a_count_v2 = 0
        a_count_t_id = 0
        a_count_v1_id = 0
        a_count_v2_id = 0

        b_temp = []
        b_temp_t_id = []
        b_temp_v1_id = []
        b_temp_v2_id = []
        b_count_t = 0
        b_count_v1 = 0
        b_count_v2 = 0
        b_count_t_id = 0
        b_count_v1_id = 0
        b_count_v2_id = 0
        # use five camera pairs
        for i in xrange(5):
            for k in xrange(f[f['labeled'][0][i]][0].size):
                print i,k
                # train dataset from labeled
                a_temp_t_id.append(k)
                b_temp_t_id.append(k)
                for j in it.chain(xrange(1, 5), xrange(6, 10)):
                    if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                        a_temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((224,224))) / 255.)
                    if len(f[f[f['detected'][0][i]][j][k]].shape) == 3:
                        b_temp.append(np.array((Image.fromarray(f[f[f['detected'][0][i]][j][k]][:].transpose(2,1,0))).resize((224,224))) / 255.)
                fwat.create_dataset(str(a_count_t),data = np.array(a_temp))
                fwbt.create_dataset(str(b_count_t),data = np.array(b_temp))
                a_temp = []
                b_temp = []
                a_count_t += 1
                b_count_t += 1

                # validation dataset from labeled
                a_temp_v1_id.append(k)
                b_temp_v1_id.append(k)
                for j in xrange(1):
                    if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                        a_temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((224,224))) / 255.)
                    if len(f[f[f['detected'][0][i]][j][k]].shape) == 3:
                        b_temp.append(np.array((Image.fromarray(f[f[f['detected'][0][i]][j][k]][:].transpose(2,1,0))).resize((224,224))) / 255.)
                fwav1.create_dataset(str(a_count_v1),data = np.array(a_temp))
                fwbv1.create_dataset(str(b_count_v1),data = np.array(b_temp))
                a_temp = []
                b_temp = []
                a_count_v1 += 1
                b_count_v1 += 1

                a_temp_v2_id.append(k)
                b_temp_v2_id.append(k)
                for j in xrange(5,6):
                    if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                        a_temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((224,224))) / 255.)
                    if len(f[f[f['detected'][0][i]][j][k]].shape) == 3:
                        b_temp.append(np.array((Image.fromarray(f[f[f['detected'][0][i]][j][k]][:].transpose(2,1,0))).resize((224,224))) / 255.)
                fwav2.create_dataset(str(a_count_v2),data = np.array(a_temp))
                fwbv2.create_dataset(str(b_count_v2),data = np.array(b_temp))
                a_temp = []
                b_temp = []
                a_count_v2 += 1
                b_count_v2 += 1


            fwat_id.create_dataset(str(a_count_t_id),data = np.array(a_temp_t_id))
            fwav1_id.create_dataset(str(a_count_v1_id),data = np.array(a_temp_v1_id))
            fwav2_id.create_dataset(str(a_count_v2_id),data = np.array(a_temp_v2_id))
            a_count_t_id += 1
            a_count_v1_id += 1
            a_count_v2_id += 1
            a_temp_t_id = []
            a_temp_v1_id = []
            a_temp_v2_id = []

            fwbt_id.create_dataset(str(b_count_t_id),data = np.array(b_temp_t_id))
            fwbv1_id.create_dataset(str(b_count_v1_id),data = np.array(b_temp_v1_id))
            fwbv2_id.create_dataset(str(b_count_v2_id),data = np.array(b_temp_v2_id))
            b_count_t_id += 1
            b_count_v1_id += 1
            b_count_v2_id += 1
            b_temp_t_id = []
            b_temp_v1_id = []
            b_temp_v2_id = []


if __name__ == '__main__':

    args = parse_args()
    create_dataset(args.mat_file_path)
