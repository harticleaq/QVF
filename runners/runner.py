import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
from envs import REGISTRY as ENV_REGISTRY
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import time

class Runner:
    def __init__(self, args, logger, config, ex_run):
        self.ex_run = ex_run
        self.config = config
        self.logger = logger
        self.args = args
        self.env = ENV_REGISTRY[self.args.env](**self.args.env_args)
        env_info = self.env.get_env_info()
        self.args.n_actions = env_info["n_actions"]
        self.args.n_agents = env_info["n_agents"]
        self.args.state_shape = env_info["state_shape"]
        self.args.obs_shape = env_info["obs_shape"]
        self.args.episode_limit = env_info["episode_limit"]
        self.args.unit_dim = 4 + self.env.shield_bits_ally + self.env.unit_type_bits
        
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(self.env, self.agents, args)
        if not args.evaluate:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        self.run_dir = self.args.run_dir
        self.log_dir = self.run_dir + '/logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if self.args.use_tensorboard:
            self.writter = SummaryWriter(self.log_dir)
        else:
            self.writter = self.ex_run
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.env_args["map_name"]
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, 0
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            # and time_steps != 0
            if (time_steps // self.args.evaluate_cycle) > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                print('win_rate is ', win_rate, "--ave reward is", episode_reward)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)
                evaluate_steps += 1
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, reward, win_tag, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                train_infos = self.agents.train(mini_batch, train_steps)
                # log
                
                # if train_steps % self.args.log_interval == 0:
                    # if self.arg.use_tensorboard:
                    #     for agent_id in range(self.args.n_agents):
                    #         for k, v in train_infos[agent_id].items():
                    #             agent_k = "agent%i/" % agent_id + k
                    #             self.writter.add_scalars(agent_k, {agent_k: v}, train_steps)
                    # else:
                    #     for agent_id in range(self.args.n_agents):
                    #         for k, v in train_infos[agent_id].items():
                    #             agent_k = "agent%i/" % agent_id + k
                    #             self.writter.log_scalar(agent_k, v, train_steps)
                train_steps += 1
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()









