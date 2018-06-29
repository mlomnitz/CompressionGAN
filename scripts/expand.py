#!/usr/bin/python3
"""
Script to run decoder inference on a single stored, quantized feature map (i.e. output of compression).
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from comp_gan.network import Network
from comp_gan.utils import Utils
from comp_gan.data import Data
from comp_gan.model import Model
from comp_gan.config import config_test, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def single_expand(config, args):
    """
    Decode a single image usign pre-trained network
    Args
    - config: Reference to the model configuration
    - args: Parsed arguments for the decodding
    Output: 
    - Recsontructed image
    TODOs:
    Assuming that conditional gan implementation is no longer in use. Might implement later
    """
    start = time.time()
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'
    assert not (config.use_conditional_GAN), 'Conditional GAN is not implemented at present'

    paths = np.array([args.compressed_path])
    gan = Model(config, paths, name='single_compress', dataset=args.dataset, evaluate=True)
    saver = tf.train.Saver()
    # missing here
    feed_dict_init = {gan.path_placeholder: paths}
    #
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        handle = sess.run(gan.train_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))
        print('Here1')
        sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_init)
        print('Here2')
        eval_dict = {gan.training_phase: False, gan.handle: handle}
        
        assert( args.compressed_path is not None), 'Input has not been specified'
        
        if args.output_path is None:
            output = os.path.splitext(os.path.basename(args.image_path))
            save_path = os.path.join(directories.samples, '{}_compressed.pdf'.format(output[0]))
        else:
            save_path = args.output_path
        Utils.decode(sess, gan, handle, args.compressed_path, save_path, config)

        print('Reconstruction saved to', save_path)

    return

def main(**kwargs):
    """
    Script main, parses arugments and runs inference on compressed file to 
    recusntruct the image
    Args:
    - Arguments to be parsed during inference
    TODOs:
    Make it work
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-c", "--compressed_path", help="path to compressed file to expand", type=str)
    parser.add_argument("-o", "--output_path", help="path to output image", type=str)
    parser.add_argument("-ds", "--dataset", default="faces", help="choice of training dataset. Currently only supports cityscapes/ADE20k/faces", choices=set(("cityscapes", "ADE20k", "faces")), type=str)
    args = parser.parse_args()

    #Launch decompression
    single_expand(config_test, args)

if __name__ == '__main__':
    main()
