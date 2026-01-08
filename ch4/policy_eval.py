if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld

def eval_onestep(pi, V, env, gamma=0.9): #env, 즉 환경은 상태전이 P 함수와 보상 R 함수를 포함함
    for state in env.states(): # 특정 시간 t에서의 모든 상태에 대해 반복
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]  # 상태에서의 행동 확률 분포
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state]) # 좌우, 위아래 이동에 대한 가치의 기댓값 계산
        
        V[state] = new_V
    return V

def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        #갱신된 양의 최대값 계산
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if t > delta:
                delta = t   

        # 임곗값과 비교
        if delta < threshold:
            break
    return V

if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9  # 할인율

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})  # 정책
    V = defaultdict(lambda: 0)  # 가치 함수

    V = policy_eval(pi, V, env, gamma)  # 정책 평가

    # 무작위 정책의 가치 함수
    env.render_v(V, pi)