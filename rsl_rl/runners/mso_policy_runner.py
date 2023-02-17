# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

from .on_policy_runner import OnPolicyRunner


from .meta_strategy_optimization.group_up_mso_optimizer import GroupUPMSOOptimizer


class MSOPolicyRunner(OnPolicyRunner): # inherit from on policy runner

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env,
                 train_cfg,
                 log_dir=log_dir,
                 device=device)

        self.mso_cfg = train_cfg["MSO"]
        self.num_envs = self.env.num_envs

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(self.env.num_obs + self.mso_cfg["UP_dim"],
                                                       self.env.num_obs + self.mso_cfg["UP_dim"], # priv num obs, same as obs
                                                       self.env.num_actions,
                                                       **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs + self.mso_cfg["UP_dim"]],
                              [self.env.num_privileged_obs], [self.env.num_actions])

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initiailize mso optimizer
        skill_optimizer = GroupUPMSOOptimizer(self.env, dim=self.mso_cfg["UP_dim"],
                                              cfg=self.mso_cfg, eval_num=1, test_steps=self.num_steps_per_env,
                                              normalized_range=True, device=self.device)  # sanity check: eval_num=100

        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        obs = obs_dict['obs'].to(self.device)
        # self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = []
        lenbuffer = []
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            with torch.inference_mode():
                if it % self.mso_cfg["optim_every_n"] == 0:
                    print("==== Construct task set ======")
                    skill_optimizer.reset()
                    skill_optimizer.policy = self.get_inference_policy(device=self.device)

                    # randomly sample and set optimizer's env params
                    self.env.resample_env_params(self.mso_cfg['group_envs'])
                    self.env.set_env_params()  # optimizer's envs are updated automatically

                    # get optimized embeddings, interact with env inside
                    skill_optimizer.optimize(maxiter=10)
                    optimized_embedding = skill_optimizer.best_x  # dim: (env_nums, x_dim)
                    optimized_embedding = torch.from_numpy(optimized_embedding).to(self.device).to(torch.float)

                # init obs for each rollout
                obs_UP = torch.cat([obs, optimized_embedding], dim=-1)

                start = time.time()
                self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
                # Rollout
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_UP, obs_UP)
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    obs = obs_dict['obs']
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    obs_UP = torch.cat([obs, optimized_embedding], dim=-1)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False) # if done, extend the rewbuffer
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_UP)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        super().log(locs, width=80, pad=35)

    def save(self, path, infos=None):
        super().save(path, infos=None)

    def load(self, path, load_optimizer=True):
        super().load(path, load_optimizer=True)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference




        #
        # return np.concatenate((task_parameters_mass_group, task_parameters_friction_group, task_parameters_alpha_group),
        #                       axis=1), \
        #        np.concatenate((task_parameters_mass, task_parameters_friction, task_parameters_alpha), axis=1)
