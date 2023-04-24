from benchbot_api import ActionResult, Agent, BenchBot

_CLASS_LIST = [
    'bottle', 'cup', 'knife', 'bowl', 'wine glass', 'fork', 'spoon', 'banana',
    'apple', 'orange', 'cake', 'potted plant', 'mouse', 'keyboard', 'laptop',
    'cell phone', 'book', 'clock', 'chair', 'table', 'couch', 'bed', 'toilet',
    'tv', 'microwave', 'toaster', 'refrigerator', 'oven', 'sink', 'person',
    'background'
]

class DemoAgent(Agent):
    def is_done(self, action_result):
        return action_result != ActionResult.SUCCESS

    def pick_action(self, observations, action_list):
        return 'move_next', {}

    def save_result(self, filename, empty_results, results_format_fns):
        pass  # No results to save yet