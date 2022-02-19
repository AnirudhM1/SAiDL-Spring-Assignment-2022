# Inspired from https://github.com/raillab/dqn
import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.agent_autoencoder import DQNAgent as DQNAgent_autoencoder
from dqn.agent_simclr import DQNAgent as DQNAgent_simclr
from dqn.agent_moco import DQNAgent as DQNAgent_moco
from dqn.agent_barlow import DQNAgent as DQNAgent_barlow
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
import torch
# torch.autograd.set_detect_anomaly(True)
import argparse

# weights and biases logger
import wandb

wandb.init(project="SIMCLR Pong", entity="anirudhm")

wandb.config = {
  "environment": "PongNoFrameskip-v4",
  "model": "moco"
}

def test(agent):
    env = gym.make("PongNoFrameskip-v4")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = gym.wrappers.Monitor(
        env, './video/test/', force=True)
    state = env.reset()
    while(True):
        action = agent.act(state)
        state,_,done,_ = env.step(action)
        if done:
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN Atari')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--set-representation', type=str, default="", help="Choosing the representation model")

    args = parser.parse_args()
    # If you have a checkpoint file, spend less time exploring
    if(args.load_checkpoint_file):
        eps_start= 0.01
    else:
        eps_start= 1

    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "dqn_type":"neurips",
        # total number of steps to run the environment for
        "num-steps": int(1e6),
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": eps_start,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 200
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])
    env.seed(hyper_params["seed"])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = gym.wrappers.Monitor(
        env, './video/', video_callable=lambda episode_id: episode_id % 100 == 0, force=True)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    if(args.set_representation == "autoencoder"):
        print('Using AutoEncoder')
        Agent = DQNAgent_autoencoder
    elif(args.set_representation == "simclr"):
        print('Using SimCLR')
        Agent = DQNAgent_simclr
    elif(args.set_representation == 'moco'):
        print('Using MOCO')
        Agent = DQNAgent_moco
    elif(args.set_representation == 'barlow'):
        print('Using Barlow Twins')
        Agent = DQNAgent_barlow
    else:
        print('Vanilla DQN')
        Agent = DQNAgent
    
    agent = Agent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params["use-double-dqn"],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        gamma=hyper_params['discount-factor'],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dqn_type=hyper_params["dqn_type"]
    )

    wandb.watch((agent.policy_network))
    wandb.define_metric("episode")
    wandb.define_metric("reward", step_metric="episode")

    if(args.load_checkpoint_file):
        print(f"Loading a policy - { args.load_checkpoint_file } ")
        agent.policy_network.load_state_dict(
            torch.load(args.load_checkpoint_file))

    eps_timesteps = hyper_params["eps-fraction"] * \
        float(hyper_params["num-steps"])
    episode_rewards = [0.0]

    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        # if t<=10000 and t%1000 == 0 : print(f't={t}')
        # if t>10000 : print(f't={t}')
        
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * \
            (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()

        if(sample > eps_threshold):
            # Exploit
            action = agent.act(state)
        else:
            # Explore
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        agent.memory.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward
        if done:
            state = env.reset()
            wandb.log({"reward": episode_rewards[-1], "episode": len(episode_rewards)})
            episode_rewards.append(0.0)

        if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
            agent.optimise_td_loss()

        if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params[
                "print-freq"] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            torch.save(agent.policy_network.state_dict(), f'checkpoint.pth')
            np.savetxt('rewards_per_episode.csv', episode_rewards,
                       delimiter=',', fmt='%1.3f')
    
    test(agent)



