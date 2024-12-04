from drone_class import create_drones
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import csv

# 드론 생성
leader, follows = create_drones(1)
print(follows)
print(follows[0].get_state())

# 로그 파일 초기화
log_file = "collision_drones-main/drone_log.csv"

# CSV 파일 헤더 작성
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", 
                     "Leader_X", "Leader_Y", "Leader_Dir_X", "Leader_Dir_Y", 
                     "Follower1_X", "Follower1_Y", 
                     "Follower1_target_position_x", "Follower1_target_position_y"])

count = 0

# 애니메이션 함수
def animate(frame):
    global count

    plt.cla()

    # 리더 위치 및 방향
    leader_position = leader.position
    leader_direction = leader.direction

    # 팔로워 위치 및 목표 위치
    follower_position = follows[0].position
    follower_target_position = follows[0].target_position

    # 리더의 방향 회전
    if count < 5:
        leader.rotate_heading(np.pi / 5)
        count += 1

    # 팔로워 업데이트
    follows[0].update_all(leader.position, leader.direction)

    # 팔로워 목표 위치로 이동
    direction_to_target = follower_target_position - follower_position
    distance_to_target = np.linalg.norm(direction_to_target)

    if count >= 5:
        follows[0].position += (direction_to_target / distance_to_target) * 0.25

    # 로그에 기록
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            frame,
            leader_position[0], leader_position[1], leader_direction[0], leader_direction[1],
            follower_position[0], follower_position[1],
            follower_target_position[0], follower_target_position[1]
        ])

    # 그래프 설정
    plt.scatter(leader.position[0], leader.position[1], color='red', label='Leader', s=200)
    plt.scatter(follower_target_position[0], follower_target_position[1], color='black', label='Target Position', s=100)
    plt.scatter(follower_position[0], follower_position[1], color='blue', label='Follower 1', s=200)
    plt.plot([leader.position[0], follower_position[0]], [leader.position[1], follower_position[1]], 'b--')
    plt.quiver(leader.position[0], leader.position[1], leader.direction[0], leader.direction[1], color='red', scale=10)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title("Drone Formation Movement without RL")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)

# 애니메이션 실행
fig = plt.figure(figsize=(6, 6))
ani = FuncAnimation(fig, animate, frames=100, interval=200)

# 플롯 표시
plt.show()

