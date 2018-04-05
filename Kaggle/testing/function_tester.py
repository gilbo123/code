'''
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.misc import imread
from scipy.misc import imsave
import os
import numpy as np


import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float) / 255.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      imsave(f, images[i, :, :, :], format='png')



def hack_image(orig_image):
  #This is where the hacking occurs
  new_image = np.sinh(orig_image)#np.zeros(orig_image.shape, np.float32)

  num_classes = 1001
  with tf.Graph().as_default():
    with slim.arg_scope(inception.inception_v3_arg_scope()):
          _, end_points = inception.inception_v3(
              orig_image, num_classes=num_classes, is_training=False)

    predicted_labels = tf.argmax(end_points['Predictions'], 1)
    print(predicted_labels)

  #for each pixel in the image
  # for pixel in orig_image:
  # 	new_image[pixel] = [0, 0, 0]
  #return new_image
  return new_image



def main(_):
  filenames =[]
  batch_shape = [	1, 299, 299, 3]
  images = np.zeros(batch_shape)
  input_dir = '/home/gilbert/Documents/Kaggle/testing/input_image'
  output_dir = '/home/gilbert/Documents/Kaggle/testing/output_image'
	
  with tf.Graph().as_default():
  	x_input = tf.placeholder(tf.float32, shape=batch_shape)
  	new_images, = tf.py_func(hack_image, [x_input], [tf.float32])
  	
  	with tf.Session() as sess:
  		for filenames, images in load_images(input_dir, batch_shape):
  			out_images = sess.run(new_images, feed_dict={x_input: images})
  			save_images(out_images, filenames, output_dir)
        #print(out_images)
  		
  					



if __name__ == '__main__':
  tf.app.run()
