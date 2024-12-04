# 팔로워 드론 수
NUM_FOLLOWS = 1
REL_POS_FOLLOWERS = [[-3.54, 3.54], [-3.54, -3.54]]
# [-3.54, 3.54] 대각선 길이 = 5

# 리더 주위
SAFE_DISTANCE = 2

# 오프셋 기준
MAX_DISTANCE_TO_OFFSET = 1

# 충돌 확정 범위
COLLISION_DISTANCE = 1.5

MAX_STEPS = 1000

TOTAL_TIMESTEPS = 5_000_000

# 학습 모델 경로
SAVE_MODEL_PATH = "trained_model"
LOAD_MODEL_PATH = "trained_model.zip"
SAVE_BEST_MODEL_PATH = "best_model/"
LOAD_BEST_MODEL_PATH = "cross_leader_with_rl/best_model/best_model.zip"
LOG_PATH = "logs/"