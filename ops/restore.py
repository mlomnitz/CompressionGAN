import tensorflow as tf
import os

def restore_chkp(ckpt_file, output_ckpt):
    """
    Restore checkpoint from available index, meta and weights files. Assumes all three are in the same directory
    Args:
    - ckpt_file: Path to tensorflow files with the included basename, i.e. gan-train_epoch15.ckpt-15.
    Output:
    Produces a copy of the three tensorflow files together with checkpoint
    """
    termination = ['.meta', '.index', '.data-00000-of-00001']
    files_ok = True
    
    for term in termination:
        if not os.path.isfile(ckpt_file+term):
            files_ok = False
    assert files_ok, 'Files are missing, can not restore'
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(ckpt_file+'.meta')
        saver.restore(sess, ckpt_file)
        saver.save(sess,output_ckpt)
