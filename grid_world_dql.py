import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import os

# 定义环境参数
GRID_SIZE = 5
ACTIONS = 5  # 上下左右和原地不动
EPISODES = 1000
BATCH_SIZE = 100
REPLAY_BUFFER_SIZE = 1000
HIDDEN_SIZE = 100
GAMMA = 0.9
LEARNING_RATE = 0.001

# 定义奖励矩阵
REWARDS = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, -1, 0, 0],
    [0, 0, -1, 0, 0],
    [0, -1, 1, -1, 0],
    [0, -1, 0, 0, 0]
])

# 在文件开头添加设备选择
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 强制使用CPU
print("使用CPU")
device = torch.device("cpu")
print(f"使用设备: {device}")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        self.to(device)  # 将模型移动到GPU
    
    def forward(self, state_action):
        return self.network(state_action)

class GridWorldEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.rewards = REWARDS
        self.reset()
    
    def reset(self):
        self.current_pos = [0, 0]
        return self._get_state()
    
    def step(self, action):
        # 0: 上, 1: 下, 2: 左, 3: 右, 4: 不动
        old_pos = self.current_pos.copy()
        
        if action == 0:  # 上
            self.current_pos[0] = max(0, self.current_pos[0] - 1)
        elif action == 1:  # 下
            self.current_pos[0] = min(self.grid_size - 1, self.current_pos[0] + 1)
        elif action == 2:  # 左
            self.current_pos[1] = max(0, self.current_pos[1] - 1)
        elif action == 3:  # 右
            self.current_pos[1] = min(self.grid_size - 1, self.current_pos[1] + 1)
        # action == 4 不动
        
        reward = self.rewards[self.current_pos[0]][self.current_pos[1]]
        done = (reward == 1)  # 到达目标
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        return self.current_pos.copy()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def normalize_state_action(state, action):
    # 归一化状态和动作，并移动到GPU
    return torch.FloatTensor([
        state[0] / (GRID_SIZE - 1),
        state[1] / (GRID_SIZE - 1),
        action / (ACTIONS - 1)
    ]).to(device)

def train(episodes=EPISODES, steps_per_episode=1000):
    print(f"开始训练 - 总episode数: {episodes}, 每个episode的步数: {steps_per_episode}")
    env = GridWorldEnv()
    dqn = DQN()  # 模型已经在初始化时移动到GPU
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    td_errors = []
    state_errors = []
    trajectories = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_trajectory = [state.copy()]
        episode_reward = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        print("正在收集经验...")
        
        for step in range(steps_per_episode):
            action = random.randint(0, ACTIONS-1)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            episode_trajectory.append(next_state.copy())
            
            if len(replay_buffer) >= BATCH_SIZE:
                if step % 100 == 0:  # 每100步打印一次进度
                    print(f"Step {step}/{steps_per_episode}, Buffer size: {len(replay_buffer)}")
                
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # 计算当前Q值
                current_q_values = torch.zeros(BATCH_SIZE, device=device)  # 移动到GPU
                for i in range(BATCH_SIZE):
                    current_q_values[i] = dqn(normalize_state_action(states[i], actions[i]))
                
                # 计算目标Q值
                target_q_values = torch.zeros(BATCH_SIZE, device=device)  # 移动到GPU
                for i in range(BATCH_SIZE):
                    if dones[i]:
                        target_q_values[i] = rewards[i]
                    else:
                        next_q = float('-inf')
                        for a in range(ACTIONS):
                            q = dqn(normalize_state_action(next_states[i], a))
                            next_q = max(next_q, q.item())
                        target_q_values[i] = rewards[i] + GAMMA * next_q
                
                # 计算损失并更新网络
                loss = nn.MSELoss()(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                td_errors.append(loss.item())
                
            state = next_state
            if done:
                print(f"Episode 提前结束，在第 {step + 1} 步到达目标")
                break
        
        trajectories.append(episode_trajectory)
        
        # 计算状态值误差
        print("计算状态值误差...")
        state_error = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                max_q = float('-inf')
                for a in range(ACTIONS):
                    q = dqn(normalize_state_action([i, j], a))
                    max_q = max(max_q, q.item())
                state_error += abs(max_q - REWARDS[i][j])
        state_errors.append(state_error)
        
        print(f"Episode {episode + 1} 完成")
        print(f"总奖励: {episode_reward:.2f}")
        print(f"当前TD误差: {td_errors[-1] if td_errors else 'N/A'}")
        print(f"当前状态值误差: {state_errors[-1]}")
    
    print("\n训练完成!")
    print(f"收集的数据统计:")
    print(f"TD误差数量: {len(td_errors)}")
    print(f"状态值误差数量: {len(state_errors)}")
    print(f"轨迹数量: {len(trajectories)}")
    print(f"最后一条轨迹长度: {len(trajectories[-1])}")
    return dqn, td_errors, state_errors, trajectories

def save_training_data(model, td_errors, state_errors, trajectories, filename_prefix='dqn'):
    """保存模型和训练数据"""
    # 保存模型
    torch.save(model.state_dict(), f'{filename_prefix}_model.pth')
    
    # 将轨迹数据转换为列表形式保存
    trajectories_list = [traj.tolist() if isinstance(traj, np.ndarray) else traj for traj in trajectories]
    
    # 保存训练数据
    np.savez(f'{filename_prefix}_training_data.npz',
             td_errors=np.array(td_errors),
             state_errors=np.array(state_errors),
             trajectories=np.array(trajectories_list, dtype=object))
    print(f"模型和训练数据已保存")

def load_training_data(filename_prefix='dqn'):
    """加载模型和训练数据"""
    # 加载模型
    model = DQN()
    model.load_state_dict(torch.load(f'{filename_prefix}_model.pth'))
    model.eval()
    
    # 加载训练数据
    data = np.load(f'{filename_prefix}_training_data.npz', allow_pickle=True)
    td_errors = data['td_errors'].tolist()
    state_errors = data['state_errors'].tolist()
    trajectories = data['trajectories'].tolist()
    
    print(f"模型和训练数据已加载")
    return model, td_errors, state_errors, trajectories

def plot_trajectory(trajectories, rewards, save_path='trajectory.png'):
    """绘制轨迹图"""
    print(f"\n绘制轨迹图:")
    print(f"轨迹数量: {len(trajectories)}")
    if len(trajectories) == 0:
        print("警告: 没有轨迹数据!")
        return
    
    plt.figure(figsize=(8, 8))
    
    # 创建热力图显示访问频率
    visit_count = np.zeros((GRID_SIZE, GRID_SIZE))
    for traj in trajectories:
        for pos in traj:
            visit_count[pos[0]][pos[1]] += 1
    
    # 使用白色背景
    plt.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap='Greys', alpha=0.1)
    
    # 标记特殊区域
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if rewards[i][j] == 1:  # 目标区域蓝色
                plt.fill([j-0.5, j+0.5, j+0.5, j-0.5], 
                        [i-0.5, i-0.5, i+0.5, i+0.5], 
                        color='blue', alpha=0.3)
            elif rewards[i][j] == -1:  # forbidden区域黄色
                plt.fill([j-0.5, j+0.5, j+0.5, j-0.5], 
                        [i-0.5, i-0.5, i+0.5, i+0.5], 
                        color='yellow', alpha=0.3)
    
    # 绘制轨迹
    for traj in trajectories:
        traj = np.array(traj)
        # 根据访问频率设置线条粗细和颜色深度
        alpha = np.clip(visit_count[traj[:-1, 0], traj[:-1, 1]] / visit_count.max(), 0.2, 1)
        linewidth = 1 + 3 * (alpha - 0.2) / 0.8
        
        for i in range(len(traj)-1):
            plt.plot([traj[i,1], traj[i+1,1]], 
                    [traj[i,0], traj[i+1,0]], 
                    color='green', alpha=alpha[i], 
                    linewidth=linewidth[i])
    
    plt.grid(True)
    plt.title('Trajectory Map')
    plt.xlim(-0.5, GRID_SIZE-0.5)
    plt.ylim(GRID_SIZE-0.5, -0.5)  # 反转y轴使得原点在左上角
    plt.savefig(save_path)
    plt.close()
    print(f"轨迹图已保存到: {save_path}")

def plot_policy(policy, rewards, save_path='policy.png'):
    """绘制策略图"""
    print(f"\n绘制策略图:")
    if policy is None or policy.size == 0:
        print("警告: 没有策略数据!")
        return
        
    plt.figure(figsize=(8, 8))
    
    # 使用白色背景
    plt.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap='Greys', alpha=0.1)
    
    # 标记特殊区域
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if rewards[i][j] == 1:
                plt.fill([j-0.5, j+0.5, j+0.5, j-0.5], 
                        [i-0.5, i-0.5, i+0.5, i+0.5], 
                        color='blue', alpha=0.3)
            elif rewards[i][j] == -1:
                plt.fill([j-0.5, j+0.5, j+0.5, j-0.5], 
                        [i-0.5, i-0.5, i+0.5, i+0.5], 
                        color='yellow', alpha=0.3)
    
    # 绘制策略箭头
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            action = policy[i][j]
            if action == 4:  # 不动
                plt.plot(j, i, 'ko', markersize=10)  # 画圆圈
            else:
                dx, dy = 0, 0
                if action == 0: dy = -0.3  # 上
                elif action == 1: dy = 0.3  # 下
                elif action == 2: dx = -0.3  # 左
                elif action == 3: dx = 0.3  # 右
                plt.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    plt.grid(True)
    plt.title('Policy Map')
    plt.xlim(-0.5, GRID_SIZE-0.5)
    plt.ylim(GRID_SIZE-0.5, -0.5)  # 反转y轴使得原点在左上角
    plt.savefig(save_path)
    plt.close()
    print(f"策略图已保存到: {save_path}")

def plot_errors(td_errors, state_errors, td_error_path='td_error.png', state_error_path='state_value_error.png'):
    """绘制误差图"""
    print(f"\n绘制误差图:")
    print(f"TD误差数据点数量: {len(td_errors)}")
    print(f"状态值误差数据点数量: {len(state_errors)}")
    
    if len(td_errors) == 0:
        print("警告: 没有TD误差数据!")
        return
    if len(state_errors) == 0:
        print("警告: 没有状态值误差数据!")
        return
    
    # TD误差图
    plt.figure(figsize=(8, 6))
    plt.plot(td_errors)
    plt.title('TD Error vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('TD Error')
    plt.grid(True)
    plt.savefig(td_error_path)
    plt.close()
    print(f"TD误差图已保存到: {td_error_path}")
    
    # 状态值误差图
    plt.figure(figsize=(8, 6))
    plt.plot(state_errors)
    plt.title('State Value Error vs Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('State Value Error')
    plt.grid(True)
    plt.savefig(state_error_path)
    plt.close()
    print(f"状态值误差图已保存到: {state_error_path}")

if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
    
    # 首先选择训练模式
    print("\n请选择训练模式：")
    print("1. 1000步训练 (默认模式)")
    print("2. 100步训练 (短训练模式)")
    mode = input("请输入选择 (1/2): ").strip()
    
    # 设置训练参数
    if mode == "2":
        episodes = 1
        steps = 100
        model_prefix = 'dqn_100steps'
        print("\n使用100步训练模式")
    else:
        episodes = EPISODES
        steps = 1000
        model_prefix = 'dqn'
        print("\n使用1000步训练模式")
    
    # 检查是否存在已保存的模型
    if os.path.exists(f'{model_prefix}_model.pth') and os.path.exists(f'{model_prefix}_training_data.npz'):
        print(f"发现已保存的{steps}步模型和训练数据，是否重新训练？(y/n)")
        choice = input().lower()
        if choice == 'n':
            dqn, td_errors, state_errors, trajectories = load_training_data(model_prefix)
        else:
            print("开始重新训练...")
            dqn, td_errors, state_errors, trajectories = train(episodes=episodes, steps_per_episode=steps)
            save_training_data(dqn, td_errors, state_errors, trajectories, model_prefix)
    else:
        print(f"未发现已保存的{steps}步模型或训练数据，开始训练...")
        dqn, td_errors, state_errors, trajectories = train(episodes=episodes, steps_per_episode=steps)
        save_training_data(dqn, td_errors, state_errors, trajectories, model_prefix)
    
    print("\n数据检查:")
    print(f"TD误差数量: {len(td_errors)}")
    print(f"状态值误差数量: {len(state_errors)}")
    print(f"轨迹数量: {len(trajectories)}")
    
    if len(td_errors) == 0 or len(state_errors) == 0 or len(trajectories) == 0:
        print("警告: 数据不完整，可能影响可视化效果")
    
    print("\n计算最优策略...")
    # 获取最优策略
    policy = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            max_q = float('-inf')
            best_action = 0
            for a in range(ACTIONS):
                q = dqn(normalize_state_action([i, j], a))
                if q.item() > max_q:
                    max_q = q.item()
                    best_action = a
            policy[i][j] = best_action
    
    print("绘制并保存结果...")
    # 分别保存各个图表，使用不同的文件名
    suffix = '_100steps' if mode == "2" else ''
    plot_trajectory(trajectories, REWARDS, f'trajectory{suffix}.png')
    plot_policy(policy, REWARDS, f'policy{suffix}.png')
    plot_errors(td_errors, state_errors, f'td_error{suffix}.png', f'state_value_error{suffix}.png')
    
    print("程序运行完成!") 