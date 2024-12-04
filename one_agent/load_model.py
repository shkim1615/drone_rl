import numpy as np
import torch
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from drone_class import Drone, LeaderDrone, FollowerDrone, create_drones
from matplotlib.animation import FuncAnimation
from env_class import DroneFormationEnv
from env_callback import PlottingCallback
import csv

# 로그 파일 초기화
log_file = "one_agent/version_3/drone_log4.csv"

# CSV 파일 헤더 작성
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", 
                     "Leader_X", "Leader_Y", "Leader_Dir_X", "Leader_Dir_Y", 
                     "Leader_speed",
                     "Follower1_X", "Follower1_Y", 
                     "Follower1_target_x", "Follower1_target_y",
                     "Follower1_speed"])


# 실행 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = DroneFormationEnv()
env.reset(seed=1)

plotting_callback = PlottingCallback()

callback_list = CallbackList([plotting_callback])

# model = PPO(
#     "MlpPolicy", 
#     env,
#     verbose=1,
#     device=device,
#     n_steps=2048, 
#     batch_size=64, 
#     learning_rate=1e-4,  # 학습률 스케줄러 적용
#     clip_range=0.2,  # 클리핑 범위를 높여 더 유연한 정책 갱신
#     gamma=0.995,  # 미래의 보상을 더 중요하게
#     gae_lambda=0.98,  # GAE 람다 값 증가
#     ent_coef=0.05,  # 탐험적 행동 강화
#     use_sde=True,  # 상태 의존 탐색 사용
#     sde_sample_freq=4,  # 더 자주 탐험적 행동을 샘플링
#     normalize_advantage=True,  # 어드밴티지 정규화
# )

loaded_model_pass = "one_agent/version_3/test.zip"
best_model_pass = "one_agent/version_3/best_model2/best_model.zip"
loaded_model = PPO.load(best_model_pass, env=env, device=device)
obs, _ = env.reset(seed=1)

plt.ioff()
plt.show()

follow_pre_position = np.copy(env.follows[0].position)

def animate(frame):
    global obs, env, follow_pre_position

    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    
    # 이전 그래픽 지우기
    plt.cla()

    # 드론 위치 다시 그리기
    plt.scatter(env.leader.position[0], env.leader.position[1], color='blue', label='Leader')
    plt.scatter(env.follows[0].position[0], env.follows[0].position[1], color='red', label='Follower 1')
    # plt.scatter(env.follows[0].target_position[0], env.follows[0].target_position[1], color='green', label='target_position')

    # 포메이션 라인 그리기
    plt.plot([env.leader.position[0], env.follows[0].position[0]], [env.leader.position[1], env.follows[0].position[1]], 'r--')

    # 리더 방향 화살표로 그리기
    plt.quiver(env.leader.position[0], env.leader.position[1], env.leader.direction[0], env.leader.direction[1], color='blue', scale=10)

    # 각 드론의 속도 계산 (속도는 벡터 크기)
    leader_speed = np.linalg.norm(env.leader.direction * env.leader_velocity)
    # follower1_speed = np.linalg.norm(action)
    
    # 팔로우 드론 속도 계산(현재 위치 - 이전 위치)
    follower1_speed = np.linalg.norm([env.follows[0].position[0] - follow_pre_position[0], env.follows[0].position[1] - follow_pre_position[1]])
    
    follow_pre_position = np.copy(env.follows[0].position)
    
    # 리더 드론 주위에 반지름 1미터 점선 원 그리기
    leader_circle = plt.Circle((env.leader.position[0], env.leader.position[1]), 1, color='blue', fill=False, linestyle='dotted')
    # plt.gca().add_artist(leader_circle)

    # 드론들의 속도를 텍스트로 표시
    plt.text(env.leader.position[0], env.leader.position[1] + 1, f"Speed: {leader_speed:.2f}", color='blue')
    plt.text(env.follows[0].position[0], env.follows[0].position[1] + 1, f"Speed: {follower1_speed:.2f}", color='red')
    
    # 충돌
    if env.follows[0].distance_to_leader < 1:
        plt.text(env.leader.position[0], env.leader.position[1] - 3, "Collision!!!", color='red', fontsize=12, ha='center')

    # 로그에 기록
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            frame,
            env.leader.position[0], env.leader.position[1],
            env.leader.direction[0], env.leader.direction[1],
            leader_speed,
            env.follows[0].position[0], env.follows[0].position[1],
            env.follows[0].target_position[0], env.follows[0].target_position[1],
            follower1_speed
        ])
    
    # 그래프 설정
    preset = 10
    plt.xlim(env.leader.position[0] - preset, env.leader.position[0] + preset)
    plt.ylim(env.leader.position[1] - preset, env.leader.position[1] + preset)
    plt.title("Drone Formation Movement with Orientation")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)
    
# 애니메이션 실행
fig = plt.figure(figsize=(6, 6))
ani = FuncAnimation(fig, animate, frames=100, interval=200)
plt.show()