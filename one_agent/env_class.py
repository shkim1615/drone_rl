import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from drone_class import create_drones
from values import NUM_FOLLOWS, MAX_DISTANCE_TO_OFFSET, MAX_STEPS, SAFE_DISTANCE, COLLISION_DISTANCE
import time

# 환경 설정
class DroneFormationEnv(gym.Env):
    def __init__(self):
        super(DroneFormationEnv, self).__init__()
        
        # 상태 공간: 리더 위치, 팔로워 위치, 리더와의 거리, 오프셋과의 거리 = 2 + 4
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # 행동 공간: 팔로워 드론의 이동 및 방향
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        
        # 리더와 팔로워 드론 초기화
        self.leader, self.follows = create_drones(NUM_FOLLOWS)
        
        self.leader_velocity = 0
        
        self.max_steps = MAX_STEPS
        self.current_step = 0
        self.terminated = False         # 강제 종료(collision)
        self.truncated = False          # 자동 종료(check steps)
        
        # 랜더링을 위한 초기 설정
        #self.fig, self.ax = plt.subplots()                                              #render
        
        self.reset()

    def reset(self, seed=None, options=None):        
        seed = np.random.randint(0, 2147483647,dtype=np.int32)
        self.leader, self.follows = create_drones(NUM_FOLLOWS)
        self.leader_velocity = 0
        self.current_step = 0
        self.terminated = False
        self.truncated = False
        
        # 초기 상태 반환 및 추가 정보 반환
        # info에 멀티 에이전트 정보를 담아서 사용하는 방법..?
        obs = self._get_obs()

        return obs, {}
    
    def _get_obs(self):
        leader_state = np.concatenate([np.array(self.leader.position).flatten()])
        
        follows_state = []
        for i in range(NUM_FOLLOWS):
            follows_state.append(np.concatenate([np.array(self.follows[i].position).flatten(), np.array([self.follows[i].distance_to_leader]), 
                                                np.array([self.follows[i].distance_to_target_position])]))
        
        obs = np.concatenate((leader_state, *follows_state))

        return obs


    def step(self, action):
        # 리더 드론 이동   
        # 리더 드론의 이동 위치를 무작위로 설정 (-1.0에서 1.0 사이의 무작위 값)
        if self.current_step % 100 == 0:
            self.leader_velocity = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
            
        # 리더 드론을 속도에 따라 위치 업데이트
        target_leader_position = np.array(self.leader.position) + np.array(self.leader_velocity)
        self.leader.move(target_leader_position)
        
        # 팔로워 드론들은 행동(action)에 따라 이동
        follw_target_position = self.follows[0].position + action
        self.follows[0].move(follw_target_position)
        
        # 팔로워 드론의 리더 포지션, 방향, 리더와의 거리, 목표 이동 위치 업데이트
        # 해당 정보를 바탕으로 리워드 계산
        for i in range(NUM_FOLLOWS):
            self.follows[i].update_all(leader_position=self.leader.position, leader_direction=self.leader.direction)
            
        self.current_step += 1
        
        # 현재 상태 반환
        obs = self._get_obs()
                
        # 보상 계산
        reward = self._calculate_reward()
        
        # 종료 조건 확인
        terminated = self._is_done()
        
        # truncated (예: 시간이 초과되었을 때 종료되는지 여부)
        truncated = self.current_step >= self.max_steps
        
        # 학습 과정
        # self.render()
        
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self):
        reward = 0
        
        reward += self.reward_distance_to_target_position()
        reward += self.penalty_distance_to_leader()
        reward += self.penalty_collision()

        return reward
    
    def penalty_distance_to_leader(self):
        penalty = 0
        if self.follows[0].distance_to_leader < SAFE_DISTANCE:
            penalty -= (SAFE_DISTANCE - self.follows[0].distance_to_leader)*0.3
        
        return penalty
    
    def penalty_collision(self):
        penalty = 0
        if self.check_collision():
            penalty -= 1
        
        return penalty
    
    def reward_distance_to_target_position(self):
        reward = 0
        penalty = 0
        if self.follows[0].distance_to_target_position < MAX_DISTANCE_TO_OFFSET:
            reward += (MAX_DISTANCE_TO_OFFSET - self.follows[0].distance_to_target_position)*0.2
        else:
            penalty -= (self.follows[0].distance_to_target_position - MAX_DISTANCE_TO_OFFSET)*0.2
        if self.follows[0].distance_to_target_position == 0:
            reward += 1
            
        return reward + penalty
    
    def check_collision(self):
        if self.follows[0].distance_to_leader < COLLISION_DISTANCE:
            return True
        else:
            return False

    def _is_done(self):
        # 충돌하면 종료
        terminated = self.check_collision()        
        
    def render(self, mode='human'):
        # 기존 그림을 지우고 새로 그림
        self.ax.clear()
        
        # 리더 드론과 팔로워 드론의 위치 그리기
        leader_pos = self.leader.position
        leader_direction = self.leader.direction  # 리더 드론의 방향 벡터
        self.ax.plot(leader_pos[0], leader_pos[1], 'ro', label='Leader')  # 리더 드론 (빨간색 원)
        
        # 방향 벡터 스케일링
        direction_scale = 20  # 스케일 값 조정
        scaled_direction = leader_direction * direction_scale
    
        self.ax.arrow(
        leader_pos[0], leader_pos[1], scaled_direction[0], scaled_direction[1], 
        head_width=2, head_length=2, fc='red', ec='red'
        )
        
        # 팔로워 드론의 위치 그리기
        for i, follow in enumerate(self.follows):
            follow_pos = follow.position
            self.ax.plot(follow_pos[0], follow_pos[1], 'bo', label=f'Follower {i+1}' if i == 0 else "")  # 팔로워 드론 (파란색 원)
            # 리더와 팔로워를 잇는 선 추가
            self.ax.plot([leader_pos[0], follow_pos[0]], [leader_pos[1], follow_pos[1]], 'g--', alpha=0.5)
            
            # 팔로워 드론의 offset과의 거리 표시
            distance_to_target = follow.distance_to_target_position
            self.ax.text(follow_pos[0], follow_pos[1] + 5, f'Dist to Offset: {distance_to_target:.2f}', color='blue')
        
        # 보상 표시
        reward = self._calculate_reward()
        self.ax.text(-90, 90, f'Reward: {reward:.2f}', color='black')
        
        # 범례와 제목 설정
        self.ax.legend()
        self.ax.set_title(f'Step: {self.current_step}')
        self.ax.set_xlim(-200, 200)
        self.ax.set_ylim(-200, 200)
        
        # 렌더링 업데이트
        plt.pause(0.001)  
        plt.draw()