from benchbot_api import BenchBot
from demo_agent import DemoAgent

if __name__ == '__main__':
    BenchBot(agent=DemoAgent()).run()