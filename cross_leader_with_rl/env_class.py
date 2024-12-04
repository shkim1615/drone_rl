import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from drone_class import create_drones
from config import NUM_FOLLOWS, MAX_DISTANCE_TO_OFFSET, MAX_STEPS, SAFE_DISTANCE, COLLISION_DISTANCE
import time

# 환경 설정
class DroneFormationEnv(gym.Env):
    def __init__(self):
        super(DroneFormationEnv, self).__init__()
        
        # 리더는 위치만 필요: 2
        # 팔로워는 위치, 리더와의 거리, 오프셋과의 거리가 필요: 4
        observation_shape = 2 + 4 * NUM_FOLLOWS
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_shape,), dtype=np.float32)
        
        # 행동 공간: 팔로워 드론의 이동
        # 팔로워 드론 당 2개
        action_shape = 2 * NUM_FOLLOWS
        self.action_space = gym.spaces.Box(low=-0.5, high=0.5, shape=(action_shape,), dtype=np.float32)
        
        # 리더와 팔로워 드론 생성
        self.leader, self.follows = create_drones(NUM_FOLLOWS)
        
        self.leader_velocity = 0
        
        self.max_steps = MAX_STEPS
        self.current_step = 0
        self.terminated = False         # 강제 종료(collision)
        self.truncated = False          # 자동 종료(check steps)
        
        # 랜더링을 위한 초기 설정
        # self.fig, self.ax = plt.subplots()                                              #render
        
        self.reset()

    def reset(self, seed=None, options=None):        
        seed = np.random.randint(0, 2147483647,dtype=np.int32)
        self.leader, self.follows = create_drones(NUM_FOLLOWS)
        self.leader_velocity = 0
        self.current_step = 0
        self.terminated = False
        self.truncated = False
        
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
        # 리더 드론의 방향을 새롭게 설정. 이후에는 변화 없음
        if self.current_step == 50:
            self.leader.direction = [-1, 0]
                
        # 팔로워 드론들은 행동(action)에 따라 이동
        # 팔로우 드론마다 액션 분배
        actions = []
        for i in range(0, NUM_FOLLOWS):
            actions.append(action[i*2:i*2+2])
        
        # 타겟 포지션을 따로 저장할 필요를 못느껴서 한번에 계산하게 만들었음
        for i in range(NUM_FOLLOWS):
            follow_move_position = self.follows[i].position + actions[i] 
            self.follows[i].move(follow_move_position)
                                 
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
        # self.render()                                                               ## render
        
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self):
        reward = 0
        
        reward += self.reward_distance_to_target_position()
        # reward += self.penalty_distance_to_leader()
        reward += self.penalty_collision()
        reward += self.reward_mission()

        return reward
    
    # 리더에게 가깝게 붙었는지 확인
    def penalty_distance_to_leader(self):
        penalty = 0
        for i in range(NUM_FOLLOWS):
            if self.follows[i].distance_to_leader < SAFE_DISTANCE:
                penalty -= 0.1
        
        return penalty
    
    # 충돌 패널티
    def penalty_collision(self):
        penalty = 0
        if self.check_collision():
            penalty += -0.5
        
        return penalty
    
    # 팔로워 드론들의 목표 지점과 현재 위치로 리워드 계산
    def reward_distance_to_target_position(self):
        reward = 0
        penalty = 0
        for i in range(NUM_FOLLOWS):
            if self.follows[i].distance_to_target_position < MAX_DISTANCE_TO_OFFSET:
                reward += 0.1
            else:
                penalty -= 0.1
            
        return reward + penalty
    
    # 리더 드론과 팔로우 드론의 충돌, 팔로워끼리 충돌
    def check_collision(self):
        collision_bool = False
        
        # 리더와 팔로우 충돌
        for i in range(NUM_FOLLOWS):
            if self.follows[i].distance_to_leader < COLLISION_DISTANCE:
                collision_bool = True
        
        return collision_bool

    def check_mission(self):
        if self.current_step > 50:
            if np.array_equal(self.follows[0].position, self.follows[0].target_position):
                return True
        
    def reward_mission(self):
        reward = 0
        if self.check_mission():
            reward += MAX_STEPS - self.current_step
        return reward 
    
    def _is_done(self):
        # 목표에 도달하면 종료
        return self.check_mission()   
    
    # 랜더 함수 실행 불가   
    # 수정 필요
    def render(self, mode='human'):
        # 기존 그림을 지우고 새로 그림
        self.ax.clear()
        
        # 리더 드론과 팔로워 드론의 위치 그리기
        leader_pos = self.leader.position
        leader_direction = self.leader.direction  # 리더 드론의 방향 벡터
        self.ax.plot(leader_pos[0], leader_pos[1], 'ro', label='Leader')  # 리더 드론 (빨간색 원)
        
        # 방향 벡터 스케일링
        direction_scale = 10  # 스케일 값 조정
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
        
        # plt.quiver(self.leader.position[0], self.leader.position[1], self.leader.direction[0], self.leader.direction[1], color='red', scale=10)
        
        # 범례와 제목 설정
        self.ax.legend()
        self.ax.set_title(f'Step: {self.current_step}')
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        
        # 렌더링 업데이트
        plt.pause(0.001)  
        plt.draw()
        
        