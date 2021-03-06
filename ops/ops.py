# -*- coding: utf-8 -*-
# Preprocessing images helper for face-compression GAN
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import numpy as np
import h5py
import pandas as pd
import subprocess
import random


def mkdir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    return

def img_preprocess( image_file, re_sample, greyscale = True):
    """
    Method used to do image pre-processing for GAN network. 
    Default sets to grayscale, fits to square canvas of size equal to the largest side
    and finally down (or upsamples) to the value given.
    """    
    image = Image.open(image_file)
    if( greyscale ):
        print('Mapping to grayscale')
        image = image.convert('L')
    side = max(image.size)    
    image = fit_to_canvas(image, side, side)
    return image.resize(re_sample )
    
def fit_to_canvas(image, desired_w, desired_h, method=Image.ANTIALIAS):
    """
    resize 'image' to 'desired_size' keeping the aspect ratio 
    and place it in center of white 'max_size' image 
    """
    old_size = image.size
    ratio_w = float(desired_w)/old_size[1]
    ratio_h = float(desired_h)/old_size[0]
    #
    delta_w = desired_w - old_size[0]
    delta_h = desired_h - old_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(image, padding)
    return new_im

def readDirToList(ds_loc, randomize = True):
    """
    build a dictionary with the paths to file in a directory
    Args:
    - location of the data set
    Output:
    - returns pandas data frame with dictionary containing the list of files
    """
    raw_dir = subprocess.Popen('find '+ds_loc+' -type f -name "*.jpg"', shell=True, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').strip()
    file_list = raw_dir.splitlines()
    if randomize:
        random.shuffle(file_list)
    return pd.DataFrame({'path':file_list})

def splitTrainValiTest(df, fraction = [70,20,10]):
    """
    split the input data frame into training, validation and test samples
    Args:
    - df: input data frame to be split
    - fraction: Percent of data to be used in training, validation and test
    Output:
    - Returns three data frames with samples for training, validation and testing
    TODOs:
    Need to randomize the lists for future use
    """
    chunk_size_base = len(df) // 10
    df_train = df[:5*chunk_size_base]
    df_vali = df[6*chunk_size_base:8*chunk_size_base]
    df_test = df[8*chunk_size_base:len(df)]
    return df_train, df_vali, df_test  

def saveToDataFrame( out_path, basename , file_list):
    """
    save the three data frames to outpath with basename 
    Args:
    - out_path : Path for output files
    - basename: File basenames for the three lists
    - file_list: list of file paths
    Output:
    - Saves the three files, returns nothing
    """
    mkdir(out_path)
    df_train, df_vali, df_test = splitTrainValiTest(file_list)
    print('Preparing lists [train, test, validation] ', len(df_train), len(df_vali), len(df_test))
    df_train.to_hdf(out_path+basename+'_train.d5','df', table=True, mode='a')
    df_test.to_hdf(out_path+basename+'_test.d5','df', table=True, mode='a')
    df_vali.to_hdf(out_path+basename+'_validation.d5','df', table=True, mode='a')
    
