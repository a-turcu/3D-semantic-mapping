from os.path import join as osj
import numpy as np
import pickle
from PIL import Image

def save_data(paths, observations, step_count):
  # Save the RGB image to file
  img_name = osj(paths['rgb'],
                          '{0:06d}.png'.format(step_count))
  Image.fromarray(observations['image_rgb']).save(img_name)

  # Save the depth image to file (numpy array format)
  depth_array = osj(paths['depth'],
                          '{0:06d}.npy'.format(step_count))
  with open(depth_array, 'wb') as f:
      np.save(f, observations['image_depth'])
  
  # Save the instance image to file (numpy array format)
  instance_array = osj(paths['instance'],
                          '{0:06d}.npy'.format(step_count))
  with open(instance_array, 'wb') as f:
      np.save(f, observations['image_segment']['instance_segment_img'])
  
  # Save the class image to file
  class_img_name = osj(paths['class'],
                          '{0:06d}.png'.format(step_count))
  Image.fromarray(observations['image_segment']['class_segment_img'].astype(np.uint8)).save(class_img_name)

  # Save all the pose data for the current pose to a pickle file
  # NOTE may want to condense to a single file saved at the end but
  # for now keeping consistent with other data points
  poses_pickle = osj(paths['poses'], '{0:06d}.pkl'.format(step_count))
  with open(poses_pickle, 'wb') as f:
      pickle.dump(observations['poses'], f)
  
  # Save all the laser data for the current pose to a pickle file
  # NOTE may want to condense to a single file saved at the end but
  # for now keeping consistent with other data points
  laser_pickle = osj(paths['laser'], 
                              '{0:06d}.pkl'.format(step_count))
  with open(laser_pickle, 'wb') as f:
      pickle.dump(observations['laser'], f)