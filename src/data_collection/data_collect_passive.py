#!/usr/bin/env python3

# Demo agent for passive mode that steps through each of the available poses in
# the environment, executing the "move_next" action recursively. Observations
# are visualised by the agent at the completion of each action.

import json
import os
from os.path import join as osj
import sys
import numpy as np
from data_save_utils import save_data
import argparse

import select
import signal
import sys

from benchbot_api import BenchBot, ActionResult, Agent
from benchbot_api.tools import ObservationVisualiser

class DataCollectPassiveAgent(Agent):

    def __init__(self, result_dir):
        self.raw_results = []
        # Root folder for all data to be saved
        self.result_dir = result_dir
        if not os.path.isdir(result_dir):
            self.run_id = 0
        else:
            self.run_id = -1

        self.step_count = 0
        
        # The environment we are currently testing in (to be filled out later)
        self.env = None

        signal.signal(signal.SIGINT, self._die_gracefully)

        self.vis = ObservationVisualiser(vis_list=['image_rgb', 'image_depth', 'laser','poses'])
    
    def _die_gracefully(self, sig, frame):
        print("")
        sys.exit(0)

    def is_done(self, action_result):            
            
        # Go forever as long as we have a action_result of SUCCESS
        return action_result != ActionResult.SUCCESS

    def pick_action(self, observations, action_list):
        # Update the visualisation
        self.vis.visualise(observations, self.step_count)

        # if it is the first observation, add an entry to the started column
        # also ensure all image folders are available
        if self.step_count == 0:
            
            if self.run_id < 0:
                self.run_id = len([d for d in os.listdir(self.result_dir) 
                                  if os.path.isdir(osj(self.result_dir, d))])
            rgb_img_path = osj(self.result_dir, self.env, 
                                '{0:06d}'.format(self.run_id), 'RGB')
        
            info_rgb_path = osj(self.result_dir, self.env, 
                                '{0:06d}'.format(self.run_id), 'RGB_info')

            depth_img_path = osj(self.result_dir, self.env, 
                                '{0:06d}'.format(self.run_id), 'depth')
            # inst_img_path = osj(self.result_dir, self.env, 
            #                     '{0:06d}'.format(self.run_id), 'instance_segment')
            # class_img_path = osj(self.result_dir, self.env, 
            #                     '{0:06d}'.format(self.run_id), 'class_segment')
            laser_path = osj(self.result_dir, self.env, 
                                '{0:06d}'.format(self.run_id), 'laser')
            poses_path = osj(self.result_dir, self.env, 
                                '{0:06d}'.format(self.run_id), 'poses')
            self.paths = {'rgb': rgb_img_path, 'depth': depth_img_path,
                          'info_rgb': info_rgb_path,
                          'laser': laser_path, 'poses': poses_path}
            
            for key, save_path in self.paths.items():    
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
            
            print("Folders ready to start saving data\nPress ENTER to begin collecting data (Ctrl^C to exit): ", end='')
            sys.stdout.flush()
            i = None
            while not i:
                i, _, _ = select.select([sys.stdin], [], [], 0)
                self.vis.update()
            sys.stdin.readline()
        
        save_data(self.paths, observations, self.step_count)

        self.step_count += 1
        return 'move_next', {} 
    
    def save_result(self, filename, empty_results, empty_object_fn):

        return
    
    def set_env(self, env_details):
        self.env = "{0}_{1}".format(env_details['name'], env_details['variant'])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', '-s', default='benchbot_data_passive', help='Location of the root folder where data will be stored')
    args = parser.parse_args()
    # Run BenchBot
    benchbot_inst = BenchBot(agent=DataCollectPassiveAgent(args.save_root))
    benchbot_inst.agent.set_env(benchbot_inst.config['environments'][0])
    benchbot_inst.run()
    