# ~/my_semantic_slam/my_semantic_slam.py

from benchbot_api import ActionResult, Agent, BenchBot
from votenet_tutorial.votenet_benchbot import votenet_build, votenet_detection  # new imports!

class MyAgent(Agent):
    def __init__(self):
        self._votenet = votenet_build()

    def is_done(self, action_result):
        return action_result != ActionResult.SUCCESS

    def pick_action(self, observations, action_list):
        results = votenet_detection(self._votenet, observations)
        print("Detected %d objects in the frame: %s" %
              (len(results), ",".join(r['class'] for r in results)))
        return 'move_next', {}

    def save_result(self, filename, empty_results, results_formant_fns):
        pass  # No results to save yet

if __name__ == '__main__':
    print("Welcome to my Semantic SLAM solution!")
    BenchBot(agent=MyAgent()).run()
