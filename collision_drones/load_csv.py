import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# 로그 파일 읽기
log_file = "collision_drones/drone_log.csv"
df = pd.read_csv(log_file)

# 애니메이션 함수 정의
fig, ax = plt.subplots(figsize=(6, 6))

def animate(frame):
    # 현재 프레임의 데이터를 가져옴
    data = df.iloc[frame]
    
    # 리더 위치와 방향
    leader_x, leader_y = data['Leader_X'], data['Leader_Y']
    leader_dir_x, leader_dir_y = data['Leader_Dir_X'], data['Leader_Dir_Y']
    
    # 팔로워 위치
    follower_x, follower_y = data['Follower1_X'], data['Follower1_Y']
    
    follower_target_position_x = data["Follower1_target_position_x"]
    follower_target_position_y = data["Follower1_target_position_y"]
    
    # 그래프 초기화
    ax.clear()
    
    # 리더와 팔로워의 위치 그리기
    ax.scatter(leader_x, leader_y, color='red', label='Leader', s=200)
    ax.scatter(follower_target_position_x, follower_target_position_y, color="black", label='Target_position', s=200)
    ax.scatter(follower_x, follower_y, color='blue', label='Follower 1', s=200)
    
    
    # 리더의 방향 벡터 그리기
    ax.quiver(leader_x, leader_y, leader_dir_x, leader_dir_y, color='red', scale=8)
    
    # 그래프 설정
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("Drone Movement Replay")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.grid(True)

# 애니메이션 실행
ani = FuncAnimation(fig, animate, frames=len(df), interval=200)

# 애니메이션 표시
plt.show()
