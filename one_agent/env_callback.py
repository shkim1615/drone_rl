import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# 학습 중 실행된 에피소드 길이 콜백 함수
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(episode_durations, show_result=False):
    plt.figure(2)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.episode_durations = []

    def _on_step(self) -> bool:
        # 매 스텝마다 호출되는 함수
        # 'dones' 배열에서 현재 환경의 종료 상태를 확인
        if self.locals['dones'][0]:
            # 에피소드가 종료되면 episode length를 기록
            episode_duration = self.locals['infos'][0].get('episode', {}).get('l', 0)
            self.episode_durations.append(episode_duration)
            plot_durations(self.episode_durations)  # 클래스 내부의 episode_durations를 전달

        return True  # 계속 학습하기 위해서는 True 반환

    def _on_training_end(self) -> None:
        # 학습이 끝났을 때 최종 결과를 표시
        plot_durations(self.episode_durations, show_result=True)
        
def plot_rewards(episode_rewards, show_result=False):
    plt.figure(2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # 최근 100 에피소드 평균 보상 그래프
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
class Rewards_PlottingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(Rewards_PlottingCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # 'dones' 배열에서 현재 환경의 종료 상태를 확인
        if self.locals['dones'][0]:
            # 에피소드가 종료되면 episode reward를 기록
            episode_reward = self.locals['infos'][0].get('episode', {}).get('r', 0)
            self.episode_rewards.append(episode_reward)
            plot_rewards(self.episode_rewards)  # 에피소드 당 보상 그래프 업데이트

        return True  # 계속 학습하기 위해서는 True 반환

    def _on_training_end(self) -> None:
        # 학습이 끝났을 때 최종 결과를 표시
        plot_rewards(self.episode_rewards, show_result=True)
 
def create_eval_callback(env):
    eval_env = env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./one_agent/version_3/best_model2/",
        log_path="./one_agent/version_3/logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    return eval_callback