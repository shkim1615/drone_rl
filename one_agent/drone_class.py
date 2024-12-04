import numpy as np
from values import REL_POS_FOLLOWERS

# 드론 클래스 정의
class Drone:
    def __init__(self, drone_id, init_position = [0, 0], init_direction = [1, 0]):
        self.drone_id = drone_id
        self.position = np.array(init_position, dtype=np.float32)  # 드론의 위치 (x, y)
        self.direction = np.array(init_direction, dtype=np.float32)
        
    def move(self, target_position):
        self.set_direction(target_position)
        self.set_position(target_position)
                
    def set_position(self, target_position):
        self.position = np.array(target_position, dtype=np.float32)
    
    def set_direction(self, target_position):
        # 목표 위치로의 방향 벡터 계산
        direction_vector = np.array(target_position) - self.position
        # print(direction_vector)
        
        # 방향 벡터의 크기를 계산
        norm = np.linalg.norm(direction_vector)
        
        # 벡터의 크기가 0이 아니면 정규화 (방향 설정), 0이면 방향 유지
        if norm != 0:
            self.direction = direction_vector / norm
    
    def get_state(self):
        return self.drone_id, self.position, self.direction


# 리더 드론
class LeaderDrone(Drone):
    def __init__(self, drone_id=0, init_position=[0, 0], init_direction=[1, 0]):
        # 부모 클래스(Drone)의 초기화 메서드를 호출하여 기본 설정을 상속받음
        super().__init__(drone_id, init_position, init_direction)
        
    def check_follow(self, follow_drone):
        pass

# 팔로워 드론
class FollowerDrone(Drone):
    def __init__(self, drone_id, init_position = [0, 0], init_direction = [1, 0], offset = [0,0]):
        super().__init__(drone_id, init_position, init_direction)
        self.offset = np.array(offset, dtype=np.float32)
        
        # 업데이트되는 항목
        self.leader_position = np.array([0, 0], dtype=np.float32)                               # 리더의 현재 위치
        self.leader_direction = np.array([1, 0], dtype=np.float32)
        self.distance_to_leader = np.linalg.norm(self.leader_position - init_position)
        self.target_position = init_position
        self.distance_to_target_position = 0
    
    # 리더 방향에 맞게 오프셋 수정 후 팔로워 드론 타겟 위치 설정
    # 본인 오프셋을 계속 사용하도록 새로 만들어서 반환
    # 리더의 위치는 0, 0으로 가정하고 방향만 돌림
    def update_target_position(self):
        angle = np.arctan2(self.leader_direction[1], self.leader_direction[0])
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array(
            [[cos_angle, -sin_angle],
            [sin_angle, cos_angle]],
            dtype=np.float32
        )
        rotated_offset = np.dot(rotation_matrix, self.offset)
        self.target_position = self.leader_position + rotated_offset
    
    def update_leader_position_and_direction(self, leader_position, leader_direction):
        self.leader_position = np.array(leader_position, dtype=np.float32)
        self.leader_direction = np.array(leader_direction, dtype=np.float32)
        
    def update_distance_to_leader(self):
        self.distance_to_leader = np.linalg.norm(self.leader_position - self.position)
        
    def update_distance_to_target_position(self):
        self.distance_to_target_position = np.linalg.norm(self.position - self.target_position)
        
    # 업데이트 순서 지켜져야 함
    def update_all(self, leader_position, leader_direction):
        self.update_leader_position_and_direction(leader_position, leader_direction)
        self.update_distance_to_leader()
        self.update_target_position()
        self.update_distance_to_target_position()
    
    # 여기에다가 action을 넣어서 방향을 맞추는 것도 학습을 하도록.    
    def set_leader_direction(self):
        self.direction = self.leader_direction

def create_drones(num_follows):
    # 리더 드론 생성
    leader_drone = LeaderDrone()
    
    # 팔로워 드론을 num_followers 수만큼 리스트에 생성
    follower_drones = [
        FollowerDrone(drone_id=i + 1, init_position=REL_POS_FOLLOWERS[i], offset=REL_POS_FOLLOWERS[i]) 
        for i in range(num_follows)
    ]
    
    return leader_drone, follower_drones