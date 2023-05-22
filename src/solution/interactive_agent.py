from __future__ import print_function

import numpy as np
import select
import signal
import sys
import open3d as o3d

from benchbot_api import ActionResult, Agent
from benchbot_api.tools import ObservationVisualiser
from mmdet3d.apis import init_model
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.registry import VISUALIZERS
from detection import detect

from registration import combine_PCD, load_pcd_pipeline
from utils_pcd import open3d_to_numpy

CHCKPOINT_FILE = "mmdetection3d/checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth"
CONFIG_FILE = "mmdetection3d/configs/votenet/votenet_8xb16_sunrgbd-3d.py"


class InteractiveAgent(Agent):

    def __init__(self):
        self.vis = ObservationVisualiser()
        self.step_count = 0
        self.combined_pcd = None
        self.model = init_model(CONFIG_FILE, CHCKPOINT_FILE)

        signal.signal(signal.SIGINT, self._die_gracefully)

    def _die_gracefully(self, sig, frame):
        print("")
        sys.exit(0)

    def is_done(self, action_result):
        # Go forever as long as we have a action_result of SUCCESS
        return action_result != ActionResult.SUCCESS

    def pick_action(self, observations, action_list):
        # Perform a sanity check to confirm we have valid actions available
        new_pcd = load_pcd_pipeline('', observations, mode='live')
        if self.step_count == 0:
            self.combined_pcd = new_pcd
            o3d.visualization.draw_geometries([self.combined_pcd])
        else:
            self.combined_pcd = combine_PCD(self.combined_pcd, new_pcd)

        if ('move_distance' not in action_list or
                'move_angle' not in action_list):
            raise ValueError(
                "We don't have any usable actions. Is BenchBot running in the "
                "right mode (active), or should it have exited (collided / "
                "finished)?")

        # Update the visualisation
        self.vis.visualise(observations, self.step_count)

        if self.step_count % 10 == 0 and self.step_count != 0:

            pcd = open3d_to_numpy(self.combined_pcd)

            pcd_path = 'pcd.npy'
            np.save(pcd_path, pcd)

            results = detect(self.model, pcd_path)

    # Prompt the user to pick an active mode action, returning when they
        # have made a valid selection
        action = None
        action_args = None
        while (action is None):
            try:
                print(
                    "Enter next action (either 'd <distance_in_metres>'"
                    " or 'a <angle_in_degrees>'): ",
                    end='')
                sys.stdout.flush()
                i = None
                while not i:
                    i, _, _ = select.select([sys.stdin], [], [], 0)
                    self.vis.update()
                action_text = sys.stdin.readline().split(" ")
                if action_text[0] == 'a':
                    action = 'move_angle'
                    action_args = {
                        'angle': (0 if len(action_text) == 1 else float(
                            action_text[1]))
                    }
                elif action_text[0] == 'd':
                    action = 'move_distance'
                    action_args = {
                        'distance':
                            0
                            if len(action_text) == 1 else float(action_text[1])
                    }
                else:
                    raise ValueError()
            except Exception as e:
                print(e)
                print("ERROR: Invalid selection")
                action = None
        self.step_count += 1
        return (action, action_args)

    def save_result(self, filename, empty_results, results_format_fns):
        # We have no results, we'll skip saving
        return
