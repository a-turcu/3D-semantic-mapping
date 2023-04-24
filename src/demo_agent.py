from benchbot_api import ActionResult, Agent, BenchBot

class DemoAgent(Agent):
    def is_done(self, action_result):
        return action_result != ActionResult.SUCCESS

    def pick_action(self, observations, action_list):
        return 'move_next', {}

    def save_result(self, filename, empty_results, results_format_fns):
        pass  # No results to save yet