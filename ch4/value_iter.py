if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld
from ch4.policy_iter import greedy_policy
# policy iteration과 다르게 value iteration은 정책 평가과 정책 개선을 분리하지 않음
# 즉 벨만 최적 방정식을 이용해 평가와 개선을 동시에 진행
# 둘의 공통점은 둘 다 환경을 정확하게 알고 있어야 한다는 점.. real world에서는 어렵기에 mc나 td 방법을 사용

def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
        
        V[state] = max(action_values)
    return V

def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        #갱신된 양의 최대값 계산
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if t > delta:
                delta = t   

        if delta < threshold:
            break
    return V

if __name__ == '__main__':
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)  # 최적 가치 함수 찾기

    pi = greedy_policy(V, env, gamma)  # 최적 정책 찾기
    env.render_v(V, pi)