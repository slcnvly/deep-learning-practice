import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # for importing the parent dirs
import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld
# 몬테 카를로의 핵심은 환경, 즉 상태전이 P 함수와 보상 R 함수를 모른다는 것
# 정책 pi와 가치 함수 V는 iterative 방법과 동일하게 초기화해서 사용
# 따라서 목표는 에이전트가 얻은 (1) 경험을 바탕으로 (2) 가치 함수 V을 증분 방식으로 추정하고 이후에 (3) 최적 정책을 찾는 방식

# 몬테카를로법을 이용해 정책평가를 수행하는 에이전트 구현
class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = [] # 에이전트가 실제로 행동하여 얻은 경험 (상태, 행동, 보상)의 리스트

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs) #probs의 확률 분포에 따라 행동을 '한 개씩' 샘플링하는게 핵심
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state] # 몬테카를로 방식의 가치 함수 갱신

env = GridWorld()
agent = RandomAgent()

episodes = 1000
for episode in range(episodes):  # 에피소드 1000번 수행
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)             # 행동 선택
        next_state, reward, done = env.step(action)  # 행동 수행

        agent.add(state, action, reward)  # (상태, 행동, 보상) 저장
        if done:   # 목표에 도달 시
            agent.eval()  # 몬테카를로법으로 가치 함수 갱신
            break         # 다음 에피소드 시작

        state = next_state

# [그림 5-12] 몬테카를로법으로 얻은 가치 함수
env.render_v(agent.V)
        