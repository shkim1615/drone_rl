# Frame,Leader_X,Leader_Y,Leader_Dir_X,Leader_Dir_Y,Leader_speed,Follower1_X,Follower1_Y,Follower1_target_x,Follower1_target_y,Follower1_speed

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# 로그 파일 읽기
log_file = "one_agent/drone_log4.csv"
df = pd.read_csv(log_file)

# 애니메이션 함수 정의
fig, ax = plt.subplots(figsize=(6, 6))

def animate(frame):
    # 현재 프레임의 데이터를 가져옴
    data = df.iloc[frame]
    
    # 리더 위치와 방향
    leader_x, leader_y = data['Leader_X'], data['Leader_Y']
    leader_dir_x, leader_dir_y = data['Leader_Dir_X'], data['Leader_Dir_Y']
    leader_speed = data["Leader_speed"]
    
    # 팔로워 위치
    follower_x, follower_y = data['Follower1_X'], data['Follower1_Y']
    
    Follower1_target_x = data["Follower1_target_x"]
    Follower1_target_y = data["Follower1_target_y"]
    follower1_speed = data["Follower1_speed"]
    
    # 그래프 초기화
    ax.clear()
    
    # 리더와 팔로워의 위치 그리기
    ax.scatter(leader_x, leader_y, color='red', label='Leader')
    # ax.scatter(Follower1_target_x, Follower1_target_y, color="black", label='Target_position')
    ax.scatter(follower_x, follower_y, color='blue', label='Follower 1')
    
    # 포메이션 라인 그리기
    plt.plot([leader_x, follower_x], [leader_y, follower_y], 'g--')
    
    
    # 리더의 방향 벡터 그리기
    ax.quiver(leader_x, leader_y, leader_dir_x, leader_dir_y, color='red', scale=10)
    
    # 속도
    plt.text(leader_x, leader_y + 1, f"Speed: {leader_speed:.2f}", color='red')
    plt.text(follower_x, follower_y + 1, f"Speed: {follower1_speed:.2f}", color='blue')
    
    # 그래프 설정
    preset = 10
    plt.xlim(leader_x - preset, leader_x + preset)
    plt.ylim(leader_y - preset, leader_y + preset)
    plt.title("Drone Formation Movement with Orientation")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)

# 애니메이션 실행
ani = FuncAnimation(fig, animate, frames=len(df), interval=200)

# 애니메이션 표시
plt.show()
