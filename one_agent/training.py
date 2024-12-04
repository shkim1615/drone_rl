import torch
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from env_class import DroneFormationEnv
from env_callback import PlottingCallback, plot_durations, create_eval_callback, Rewards_PlottingCallback, plot_rewards

# 실행 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = DroneFormationEnv()
seed = np.random.randint(0, 2147483647,dtype=np.int32)
env.reset(seed=seed)

plotting_callback = PlottingCallback()

eval_callback = create_eval_callback(env=env)

rewards_plotting_callback = Rewards_PlottingCallback()

callback_list = CallbackList([rewards_plotting_callback, eval_callback])

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

# model = PPO("MlpPolicy", env, verbose=1, device=device)

model = PPO.load("one_agent/version_3/best_model1/best_model.zip", env=env, device=device)

model.learn(total_timesteps=50_000_000, callback=callback_list)

model.save("one_agent/version_3/test")

print('Complete')
plot_durations(plotting_callback.episode_durations, show_result=True)
plt.ioff()
plt.show()