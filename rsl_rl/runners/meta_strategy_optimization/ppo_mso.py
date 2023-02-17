import numpy as np
import time
# from .up_mso_optimizer import UPMSOOptimizer
from .group_up_mso_optimizer import GroupUPMSOOptimizer
import random
import torch


###############################################################
### Evaluate the performance of one sample params (all envs)###
###############################################################
def MSO_traj_generator(agent, env, task_embeddings, horizon):
    # set env and obs info before reset
    env.set_info({'set_task_embeddings': task_embeddings})
    obs = env.reset()
    ep_len = np.zeros(shape=env.num_envs)
    reward_ll_sum = 0
    done_sum = 0

    # **************************** for reward debugging **************************** #
    reach_vel_reward_sum = 0.
    input_reward_tele_sum = 0.
    input_rate_reward_boom_sum, input_rate_reward_dipper_sum, input_rate_reward_tele_sum, input_rate_reward_pitch_sum = 0., 0., 0., 0.
    # **************************** for reward debugging **************************** #


    for step in range(horizon):  # collect horizon length data before update the policy
        actor_obs = obs
        critic_obs = obs
        action = agent.observe(actor_obs)                # get the action prediction given observation; assign self.actions_log_prob
        obs, reward, dones, infos = env.step(action, False)  # obs: (env_dim, obs_dim);  dones: (env_dim,)
        ep_len[~dones] += 1
        ep_len[dones] = 0                                # If not end, ep len +1; if end ep_len is 0
        agent.step(value_obs=critic_obs, rews=reward, dones=dones, infos=[])  # predict value function V(s) and save transitions
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)

        # **************************** for reward debugging **************************** #
        reach_vel_reward_sum += sum(infos['reachVelReward'])

        input_reward_tele_sum += sum(infos['inputRewardTele'])

        input_rate_reward_boom_sum += sum(infos['inputRateRewardBoom'])
        input_rate_reward_dipper_sum += sum(infos['inputRateRewardDipper'])
        input_rate_reward_tele_sum += sum(infos['inputRateRewardTele'])
        input_rate_reward_pitch_sum += sum(infos['inputRateRewardPitch'])
        # **************************** for reward debugging **************************** #

    return obs, ep_len, reward_ll_sum, done_sum, \
           reach_vel_reward_sum, \
           input_reward_tele_sum, \
           input_rate_reward_boom_sum, input_rate_reward_dipper_sum, input_rate_reward_tele_sum, input_rate_reward_pitch_sum


def set_env_params(optimizer, cfg):
    # get task params for all envs from yaml. Only need to change the optimizer's env
    group_nums = int(cfg['environment']['num_envs'] / cfg['environment']['group_envs'])

    # SCALE
    task_parameters_scale_range = np.array(cfg['environment']['actuatorModel']['LUTResampleRange'])
    task_parameters_scale_group = np.ones(shape=(group_nums, 4), dtype=np.float32)  # row major matrix. dim: (ngroups, 4)
    task_parameters_scale_group[:, 0] = np.array([random.uniform(task_parameters_scale_range[0], task_parameters_scale_range[1]) for _ in range(group_nums)])
    task_parameters_scale_group[:, 1] = np.array([random.uniform(task_parameters_scale_range[2], task_parameters_scale_range[3]) for _ in range(group_nums)])
    task_parameters_scale_group[:, 2] = np.array([random.uniform(task_parameters_scale_range[4], task_parameters_scale_range[5]) for _ in range(group_nums)])
    task_parameters_scale_group[:, 3] = np.array([random.uniform(task_parameters_scale_range[6], task_parameters_scale_range[7]) for _ in range(group_nums)])

    task_parameters_scale = np.ones(shape=(cfg['environment']['num_envs'], 4), dtype=np.float32)  # row major matrix. dim: (env_nums, 4)
    for i in range(group_nums):
        task_parameters_scale[i * cfg['environment']['group_envs']: (i + 1) * cfg['environment']['group_envs']] = task_parameters_scale_group[i]
    optimizer.env.set_info({'set_env_UP': task_parameters_scale})  # set info for all envs by calling individual setInfo in every env

    # DELAY
    task_parameters_delay_range = np.array(cfg['environment']['actuatorModel']['LUTDelayRange'])
    task_parameters_delay_group = np.ones(shape=(group_nums, 4), dtype=np.float32)  # row major matrix. dim: (ngroups, 4)
    task_parameters_delay_group[:, 0] = np.array([random.uniform(0, task_parameters_delay_range[0]) for _ in range(group_nums)])
    task_parameters_delay_group[:, 1] = np.array([random.uniform(0, task_parameters_delay_range[1]) for _ in range(group_nums)])
    task_parameters_delay_group[:, 2] = np.array([random.uniform(0, task_parameters_delay_range[2]) for _ in range(group_nums)])
    task_parameters_delay_group[:, 3] = np.array([random.uniform(0, task_parameters_delay_range[3]) for _ in range(group_nums)])

    task_parameters_delay = np.ones(shape=(cfg['environment']['num_envs'], 4), dtype=np.float32)  # row major matrix. dim: (env_nums, 4)
    for i in range(group_nums):
        task_parameters_delay[i * cfg['environment']['group_envs']: (i + 1) * cfg['environment']['group_envs']] = task_parameters_delay_group[i]
    optimizer.env.set_info({'set_env_UP_delay': task_parameters_delay})  # set info for all envs by calling individual setInfo in every env

    # ALPHA(LOW-PASS FILTER)
    task_parameters_alpha_range = np.array(cfg['environment']['actuatorModel']['LUTAlphaRange'])
    task_parameters_alpha_group = np.ones(shape=(group_nums, 4), dtype=np.float32)  # row major matrix. dim: (ngroups, 4)
    task_parameters_alpha_group[:, 0] = np.array([random.uniform(task_parameters_alpha_range[0], task_parameters_alpha_range[1]) for _ in range(group_nums)])
    task_parameters_alpha_group[:, 1] = np.array([random.uniform(task_parameters_alpha_range[2], task_parameters_alpha_range[3]) for _ in range(group_nums)])
    task_parameters_alpha_group[:, 2] = np.array([random.uniform(task_parameters_alpha_range[4], task_parameters_alpha_range[5]) for _ in range(group_nums)])
    task_parameters_alpha_group[:, 3] = np.array([random.uniform(task_parameters_alpha_range[6], task_parameters_alpha_range[7]) for _ in range(group_nums)])

    task_parameters_alpha = np.ones(shape=(cfg['environment']['num_envs'], 4), dtype=np.float32)  # row major matrix. dim: (env_nums, 4)
    for i in range(group_nums):
        task_parameters_alpha[i * cfg['environment']['group_envs']: (i + 1) * cfg['environment']['group_envs']] = task_parameters_alpha_group[i]
    optimizer.env.set_info({'set_env_UP_alpha': task_parameters_alpha})  # set info for all envs by calling individual setInfo in every env

    return np.concatenate((task_parameters_scale_group, task_parameters_delay_group, task_parameters_alpha_group), axis=1), \
           np.concatenate((task_parameters_scale, task_parameters_delay, task_parameters_alpha), axis=1)


# pass the agent as input, actor can be accessed by agent.actor
def ppo_optimize(env, agent, cfg, eval_epoch, each_rollout_steps, skilldim=3, cfg_saver=None): # skilldim is the optimization variable dimension
    # pi = agent.actor
    total_time = 0.0

    # Used to construct the task-embedding mapping
    # skill_optimizer = UPMSOOptimizer(env, agent.actor, dim=skilldim, cfg=cfg, eval_num=5, test_steps=each_rollout_steps)
    skill_optimizer = GroupUPMSOOptimizer(env, agent.actor, dim=skilldim,
                                          cfg=cfg, eval_num=1, test_steps=each_rollout_steps,
                                          normalized_range=True) # sanity check: eval_num=100

    episodes_so_far = 0
    for update in range(cfg['algorithm']['total_algo_updates']): # algorithm iteration
        start = time.time()
        obs = env.reset()   # reset the env when start a new training episode; resample a new UP

        if update % cfg['environment']['eval_every_n'] == 0: # for evaluation every 'eval_every_n' steps
            env.show_window()
            if (cfg['environment']['record_video']):
                env.start_recording_video(cfg_saver.data_dir + "/" + str(update) + ".mp4")
            for step in range(1 * each_rollout_steps):
                action_ll, _ = agent.actor.sample(torch.from_numpy(obs).to(agent.device))
                t = time.time()
                obs, reward_ll, dones, _ = env.step(action_ll.cpu().detach().numpy(), True)
                # print(time.time()-t)

            agent.save_training(cfg_saver.data_dir, update, update)
            obs = env.reset()   # reset the env after finishing the evaluation episode
            if (cfg['environment']['record_video']):
                env.stop_recording_video()
            env.hide_window()

        if update % cfg['environment']['optim_every_n'] == 0: # tasks optimization
            print("==== Construct task set ======")
            optimized_embedding = None
            skill_optimizer.reset()
            skill_optimizer.policy = agent.actor  # update optimizer's policy

            # randomly sample and set optimizer's env params
            task_parameters_group, task_parameters = set_env_params(optimizer=skill_optimizer, cfg=cfg)

            # get optimized embeddings
            skill_optimizer.optimize(maxiter=30, max_steps=400000)
            optimized_embedding = skill_optimizer.best_x  # dim: (env_nums, x_dim)
            task_embeddings = np.concatenate((task_parameters, optimized_embedding), axis=1) # concatenation dim: (env_nums, 4+x_dim)

            print("tasks for groups: ", task_parameters_group)
            print("optimize for groups: ", optimized_embedding[::cfg['environment']['group_envs']])
            # Debug: when optimize range = real parameter range
            # print("Error in total: ", np.sum(abs(task_parameters_group - optimized_embedding[::cfg['environment']['group_envs']])) )

            # *************************************************************************** #
            # ************************** For sanity check plot ************************** #
            # *************************************************************************** #
            # sol_list_all = np.array(skill_optimizer.solution_history)
            # value_list_all = np.array(skill_optimizer.best_fitness_history)
            #
            # # # for sanity check(when nenvs = 1)
            # # sol_list_all = sol_list_all[:, np.newaxis, :]
            # # value_list_all = value_list_all[:, np.newaxis]
            #
            # for i in range(group_nums):
            #     plot_param_evolution(sol_list=sol_list_all[:,i,:], value_list = value_list_all[:, i], target_UP=task_parameters_group[i])
            # *************************************************************************** #
            # ************************** For sanity check plot ************************** #
            # *************************************************************************** #


        # setting iteration to current iteration, for curriculum
        training_iteration_vec = np.zeros(shape=(cfg['environment']['num_envs'], 1), dtype=np.float32)
        training_iteration_vec[:] = update
        env.set_info({'training_iteration': training_iteration_vec})
        err_reward_coeff_vec = np.zeros(shape=(cfg['environment']['num_envs'], 1), dtype=np.float32)
        err_reward_coeff_vec[:] = 1.
        env.set_info({'err_reward_coeff': err_reward_coeff_vec})

        # **************************** for reward debugging **************************** #
        obs, ep_len, reward_ll_sum, done_sum, \
        reach_vel_reward_sum, \
        input_reward_tele_sum, \
        input_rate_reward_boom_sum, input_rate_reward_dipper_sum, input_rate_reward_tele_sum, input_rate_reward_pitch_sum \
            = MSO_traj_generator(agent, env, task_embeddings, each_rollout_steps)
        # **************************** for reward debugging **************************** #


        agent.update(actor_obs=obs,  # update the policy
                     value_obs=obs,
                     log_this_iteration=update % 5 == 0,
                     update=update)

        end = time.time()

        total_steps_per_episode = each_rollout_steps * cfg['environment']['num_envs']  # 'num_envs' envs are created in an episode, total steps in all envs
        average_ll_performance = reward_ll_sum / total_steps_per_episode    # average reward per step per environment
        average_dones = done_sum / total_steps_per_episode
        # avg_rewards.append(average_ll_performance)
        avg_ep_leng = ep_len.mean()

        # **************************** for reward debugging **************************** #
        average_reach_vel_reward = reach_vel_reward_sum / total_steps_per_episode

        average_input_reward_tele = input_reward_tele_sum / total_steps_per_episode

        average_input_rate_reward_boom = input_rate_reward_boom_sum / total_steps_per_episode
        average_input_rate_reward_dipper = input_rate_reward_dipper_sum / total_steps_per_episode
        average_input_rate_reward_tele = input_rate_reward_tele_sum / total_steps_per_episode
        average_input_rate_reward_pitch = input_rate_reward_pitch_sum / total_steps_per_episode
        # **************************** for reward debugging **************************** #

        elapsed_ep_time = end - start
        total_time += elapsed_ep_time

        agent.writer.add_scalar('Policy/average_reward', average_ll_performance, update)
        agent.writer.add_scalar('Policy/average_dones', average_dones, update)
        agent.writer.add_scalar('Training/elapsed_time_episode', end - start, update)
        agent.writer.add_scalar('Training/fps', total_steps_per_episode / (end - start), update)
        agent.writer.add_scalar('Policy/avg_ep_len', avg_ep_leng, update)

        # **************************** for reward debugging **************************** #
        agent.writer.add_scalar('Policy/average_reach_vel_reward', average_reach_vel_reward, update)

        agent.writer.add_scalar('Policy/average_input_reward_tele', average_input_reward_tele, update)

        agent.writer.add_scalar('Policy/average_input_rate_reward_boom', average_input_rate_reward_boom, update)
        agent.writer.add_scalar('Policy/average_input_rate_reward_dipper', average_input_rate_reward_dipper, update)
        agent.writer.add_scalar('Policy/average_input_rate_reward_tele', average_input_rate_reward_tele, update)
        agent.writer.add_scalar('Policy/average_input_rate_reward_pitch', average_input_rate_reward_pitch, update)
        # **************************** for reward debugging **************************** #

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("total time [min]: ", '{:6.0f}'.format(total_time / 60.)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("avg_ep_len: ", '{:0.6f}'.format(avg_ep_leng)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(elapsed_ep_time)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_per_episode / elapsed_ep_time)))
        print('----------------------------------------------------\n')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def plot_param_evolution(sol_list, value_list, target_UP):
    plt.figure()
    plt.plot(np.arange(sol_list.shape[0]), sol_list[:, 0], label='Optimal Boom')
    plt.plot(np.arange(sol_list.shape[0]), sol_list[:, 1], label='Optimal Dipper')
    plt.plot(np.arange(sol_list.shape[0]), sol_list[:, 2], label='Optimal Tele')
    plt.plot(np.arange(sol_list.shape[0]), sol_list[:, 3], label='Optimal Pitch')
    plt.plot(np.arange(sol_list.shape[0]), target_UP[0] * np.ones(sol_list.shape[0]), label='GT Boom', linestyle="--")
    plt.plot(np.arange(sol_list.shape[0]), target_UP[1] * np.ones(sol_list.shape[0]), label='GT Dipper', linestyle="--")
    plt.plot(np.arange(sol_list.shape[0]), target_UP[2] * np.ones(sol_list.shape[0]), label='GT Tele', linestyle="--")
    plt.plot(np.arange(sol_list.shape[0]), target_UP[3] * np.ones(sol_list.shape[0]), label='GT Pitch', linestyle="--")
    plt.legend()

    plt.figure()
    plt.plot(np.arange(value_list.shape[0]), value_list, label='Fitness')
    plt.legend()


    save_path = '/home/zerenluo/rslgym_ws/m545_online_learning/scripts/runs/m545_mso_ppo/'
    filename = save_path + '/plots.pdf'
    multipage(filename)
    os.system('xdg-open ' + filename)
