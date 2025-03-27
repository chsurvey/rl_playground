import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import random

# ----------------------------------------------------------------------------
# (1) ActorCriticNet, A2CAgent 등 기존 코드 (질문에서 주신 내용) ----------------
# ----------------------------------------------------------------------------

def plot_training_metrics(scores, actor_losses, critic_losses, next_state_losses, grad_norms):
    episodes = range(len(scores))
    
    plt.figure(figsize=(15, 10))
    
    # 보상
    plt.subplot(2, 2, 1)
    plt.plot(episodes, scores, label='Score')
    plt.plot(episodes, [np.mean(scores[max(0, i-100):i+1]) for i in episodes], 
             label='Moving Avg (100)', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Episode Score")
    plt.legend()
    plt.grid(True)
    
    # 손실
    plt.subplot(2, 2, 2)
    plt.plot(episodes, actor_losses, label='Actor Loss')
    plt.plot(episodes, critic_losses, label='Critic Loss')
    plt.plot(episodes, next_state_losses, label='Next State Loss')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend()
    plt.grid(True)
    
    # 그래디언트 노름
    plt.subplot(2, 2, 3)
    plt.plot(episodes, grad_norms, label='Gradient Norm')
    plt.xlabel("Episode")
    plt.ylabel("Norm")
    plt.title("Gradient Norm per Episode")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("actor_critic_training_metrics.png")
    plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=24):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Next z predictor
        self.next_state = nn.Linear(hidden_dim, hidden_dim)
        self.state_scalar = nn.Sequential(
            nn.Linear(hidden_dim*2+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.state_decode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_size)
        )
        
        # Actor
        self.actor_head = nn.Linear(hidden_dim * 2, action_size)

        # Critic
        self.critic_head = nn.Linear(hidden_dim*2, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        z = F.relu(self.fc2(x))  # current z
        
        z_vector = F.relu(self.next_state(z))
        value = self.critic_head(torch.cat([z, z_vector], dim=1))
        z_vector = F.normalize(z_vector, p=2, dim=1)
        scalar = F.sigmoid(self.state_scalar(torch.cat([z, z_vector, value], dim=1)))
        # scalar = scalar.repeat(1, z.size(1))
        next_z = z + z_vector * scalar
        decoded_next_state = self.state_decode(next_z)
        logits = self.actor_head(torch.cat([z, next_z], dim=1))
        
        return logits, value, next_z, z_vector, decoded_next_state

    def get_state_embedding(self, x):
        with torch.no_grad():
            x = F.relu(self.fc1(x))
            z = F.relu(self.fc2(x))
            return z
        
    def get_value(self, x):
        with torch.no_grad():
            x = F.relu(self.fc1(x))
            z = F.relu(self.fc2(x))
            return self.critic_head(z)


class A2CAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.1, lr=1e-3):
        self.gamma = gamma
        self.model = ActorCriticNet(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.action_size = action_size
        self.epsilon = epsilon
    
    def get_action(self, state):
        """
        Returns an action following an epsilon-greedy strategy.
        
        With probability epsilon, it selects a random action.
        Otherwise, it samples according to the policy's distribution.
        """
        # Either pick a random action (exploration) or follow the policy (exploitation)
        if random.random() < self.epsilon:
            # Random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: follow the policy
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, _, _, _, _ = self.model(state_t)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
            return action
    
    def train_episode(self, states, actions, rewards, next_states):
        returns = self._discounted_returns(rewards)
        
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)

        logits, values, next_z, z_vector, decoded_next_state = self.model(states_t)
        values = values.squeeze(1)
        
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_actions = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        advantages = returns_t - values
        next_state_z = self.model.get_state_embedding(next_states_t)
        state_diff = -F.mse_loss(next_z, next_state_z)
        
        next_state_value = self.model.critic_head(next_z)
        next_state_loss = -torch.mean(returns_t - next_state_value)
        decoder_loss = F.mse_loss(decoded_next_state, next_states_t)
        
        actor_loss = -torch.mean(log_probs_actions * state_diff.detach())
        critic_loss = torch.mean(advantages**2)
        
        loss = actor_loss + critic_loss + next_state_loss + decoder_loss
        
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), next_state_loss.item(), grad_norm

    def _discounted_returns(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0.0
        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + self.gamma * running_sum
            discounted[t] = running_sum
        return discounted


# ----------------------------------------------------------------------------
# (2) 구간별로 최대 30회 리플레이 저장을 위한 추가 코드 -----------------------
# ----------------------------------------------------------------------------

def get_score_category(score: float):
    """점수(score)에 따라 5개 구간 이름을 리턴."""
    if score < 50:
        return '0_50'
    elif score < 200:
        return '50_200'
    elif score < 300:
        return '200_300'
    elif score < 500:
        return '300_500'
    else:
        return '500_'


def record_episode_replay(env_name, episode_seed, actions, video_folder, name_prefix=""):
    """
    학습 시 기록한 (시드, 액션 시퀀스)를 바탕으로, 
    같은 시드로 env를 reset 한 뒤 같은 액션을 재실행하며 영상을 저장.
    
    - env_name: 예) "CartPole-v1"
    - episode_seed: 학습 때 사용했던 시드
    - actions: 학습 때 실제로 수행했던 액션 시퀀스(list or np.array)
    - video_folder: mp4 파일을 저장할 폴더
    - name_prefix: 동영상 파일명에 들어갈 prefix (str)
    """
    rec_env = gym.make(env_name, render_mode="rgb_array")
    
    # name_prefix에 episode 번호, score 등 원하는 정보를 담아 MP4 파일명에 반영
    rec_env = gym.wrappers.RecordVideo(
        rec_env, 
        video_folder=video_folder, 
        name_prefix=name_prefix,
        episode_trigger=lambda ep_i: True  # 현재 1개의 에피소드만 녹화
    )
    
    obs, _ = rec_env.reset(seed=episode_seed)
    done = False
    step_idx = 0
    max_steps = len(actions)  # 학습 때 기록해둔 액션 개수만큼만 step
    
    while not done and step_idx < max_steps:
        action = actions[step_idx]
        step_idx += 1
        
        obs, reward, terminated, truncated, info = rec_env.step(action)
        done = terminated or truncated
    
    rec_env.close()


def train_actor_critic_cartpole_with_replays(env_name="CartPole-v1", episodes=1000):
    """Score 구간별로 최대 30회씩 영상 저장을 포함한 학습함수 예시."""
    env = gym.make(env_name)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = A2CAgent(state_size, action_size, gamma=0.99, epsilon=0.01, lr=1e-3)
    
    # 구간별 녹화횟수를 세는 딕셔너리
    category_counters = {
        '0_50': 0,
        '50_200': 0,
        '200_300': 0,
        '300_500': 0,
        '500_': 0
    }
    max_replays_per_section = 30
    
    scores = []
    actor_losses = []
    critic_losses = []
    next_state_losses = []
    grad_norms = []
    
    for e in range(episodes):
        # 매 에피소드마다 시드를 랜덤하게 정해둔다.
        episode_seed = np.random.randint(100000)  
        state, _ = env.reset(seed=episode_seed)
        
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        total_reward = 0
        
        while not done:
            action = agent.get_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            state = next_state
            total_reward += reward
        
        # 마지막에 'next_states'가 필요하므로, shift해서 만듦
        episode_next_states = episode_states[1:] + [state]
        
        # 학습
        actor_loss, critic_loss, next_state_loss, grad_norm = agent.train_episode(
            states=np.array(episode_states, dtype=np.float32),
            actions=np.array(episode_actions),
            rewards=np.array(episode_rewards, dtype=np.float32),
            next_states=np.array(episode_next_states, dtype=np.float32)
        )
        
        scores.append(total_reward)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        next_state_losses.append(next_state_loss)
        grad_norms.append(grad_norm)
        
        avg_score = np.mean(scores[-100:])
        
        # Score 구간 파악
        category = get_score_category(total_reward)
        
        # 해당 구간의 녹화 개수가 30 미만이면 동영상 저장
        if category_counters[category] < max_replays_per_section and e%100 == 0:
            # 파일명에 episode 번호, 실제 score를 기록
            name_prefix = f"ep_{e}_score_{total_reward:.1f}"
            video_folder = f"videos/{category}"
            
            record_episode_replay(
                env_name=env_name,
                episode_seed=episode_seed,
                actions=episode_actions,
                video_folder=video_folder,
                name_prefix=name_prefix
            )
            
            category_counters[category] += 1
        
        if e % 100 == 0:
            print(f"[Episode {e}] Score: {total_reward}, "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
                  f"Next State Loss: {next_state_loss:.4f}, Grad Norm: {grad_norm:.4f}, "
                  f"Avg(100): {avg_score:.2f}")
        
        # 간단한 CartPole 완수 조건
        if avg_score >= 400.0 and len(scores) >= 100:
            print(f"최근 100 에피소드 평균이 400 이상이므로 학습을 종료합니다. (episode: {e})")
            break
    
    env.close()
    plot_training_metrics(scores, actor_losses, critic_losses, next_state_losses, grad_norms)
    return agent


if __name__ == "__main__":
    trained_agent = train_actor_critic_cartpole_with_replays(
        env_name="CartPole-v1",
        episodes=50000
    )