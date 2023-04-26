import numpy as np
import json

def save_data(observation):

    '''
    numpy.ndarray(shape=(H,W), dtype='float32')
    '''
    image_depth = observation['image_depth']

    '''
    {
    'frame_id': string
    'height': int
    'width': int
    'matrix_instrinsics':
    numpy.ndarray(shape=(3,3),
    dtype='float64')
    'matrix_projection':
    numpy.ndarray(shape=(3,4)
    dtype='float64')
    }
    '''
    image_depth_info = observation['image_depth_info']

    '''
    numpy.ndarray(shape=(H,W, 3), dtype='uint8')
    '''
    image_rgb = observation['image_rgb']
    '''
    same as depth???
    {
    'frame_id': string
    'height': int
    'width': int
    'matrix_instrinsics':
    numpy.ndarray(shape=(3,3),
    dtype='float64')
    'matrix_projection':
    numpy.ndarray(shape=(3,4)
    dtype='float64')
    '''
    image_rgb_info = observation['image_rgb_info']


    #TODO: what other poses are there?
    camera_pose = observation['poses']['camera']


    # save rgb and depth to same npy file
    stack = np.stack((image_depth, image_rgb))
    # open file
    f = open('data.npy', 'ab')
    # save to file
    np.save(f, stack)
    # close file
    f.close()

    # save info to json file
    f = open('info.txt', 'a')
    json.dump(image_depth_info, f)
    f.close()

    # save camera pose to json file
    f = open('camera_pose.txt', 'a')
    json.dump(camera_pose, f)
    f.close()
    