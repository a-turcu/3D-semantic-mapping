#!/usr/bin/env python3

from benchbot_api import BenchBot
from interactive_agent import InteractiveAgent

if __name__ == '__main__':
    BenchBot(agent=InteractiveAgent()).run()
