# this optimizer is able to deal with multiple environment optimization
import time

from .mso_bayesian_optimization import *
import numpy as np
import torch
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sklearn.gaussian_process as gp
# import concurrent.futures

from isaacgym.torch_utils import *

class GroupUPMSOOptimizer:
    def __init__(self, env, dim, cfg, eval_num = 2, test_steps=0, normalized_range=False, scenario=None, device='cpu'): # dim is latent variable dimension, smaller than dyna param
        self.env = env
        self.dim = dim
        self.cfg = cfg
        self.device = device
        self.num_envs = self.env.num_envs
        self.num_groups = int(self.env.num_envs/ self.cfg['group_envs'])
        print("*** {} envs are divided into {} group ***".format(self.env.num_envs, self.num_groups))
        self.eval_num = eval_num
        self.normalized_range = normalized_range
        self.scenario = scenario

        # ************ bayesian optimization ************
        self.solution_history = []
        self.best_fitness_history = []
        self.best_f = 100000
        self.best_x = np.zeros((self.dim, ))  # best_x dim: (ngroup, x_dim)
        self.test_steps = test_steps

        self.random_search = 1000
        self.use_ucb = True
        self.epsilon = 1e-7
        self.alpha = 1e-2

        self.next_sample_all = np.zeros((self.num_envs, self.dim))  # all envs
        self.next_sample_group = np.zeros((self.num_groups, self.dim))  # all groups

        # Create the GP using sklearn
        kernel = gp.kernels.Matern(length_scale_bounds=(1e-2, 1e3)) + gp.kernels.WhiteKernel()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=self.alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)
        self.models = [model] * self.num_envs

        if not self.normalized_range:
            # SO real get the optimization bound from yaml, change it into many envs's bound(non-normalized).
            self.custom_bound_per_env = np.array(self.cfg['environment']['actuatorModel']['LUTResampleRange']
                                                # + [0., self.cfg['environment']['actuatorModel']['LUTDelayRange'][0],
                                                #    0., self.cfg['environment']['actuatorModel']['LUTDelayRange'][1],
                                                #    0., self.cfg['environment']['actuatorModel']['LUTDelayRange'][2],
                                                #    0., self.cfg['environment']['actuatorModel']['LUTDelayRange'][3]]
                                                + self.cfg['environment']['actuatorModel']['LUTAlphaRange'])
            self.custom_bound_per_env = np.reshape(self.custom_bound_per_env, (self.dim, 2))
        else:
            self.custom_bound_per_env_lower = -1.
            self.custom_bound_per_env_upper = 1.
            # MSO uniform range trial(normalized)
            self.custom_bound_per_env = np.tile(np.array([self.custom_bound_per_env_lower, self.custom_bound_per_env_upper]), (self.dim, 1))
        print(self.custom_bound_per_env)

        # self.concurr = ConcurrVar(self.cfg, self.dim, self.custom_bound_per_env)
        self.mu_history, self.sigma_history, self.next_sample_history, self.cv_score_history = [], [], [], []

        self.test_mode = self.scenario is not None and self.num_envs == 1
        np.random.seed(None) # make sure random sampling

    def reset(self):
        self.best_f = 100000
        self.best_x = np.zeros((self.dim, ))
        self.solution_history = []
        self.best_fitness_history = []

        # TODO: tune the parameter that is helpful for the optimization / tune optimization objective function
        # # set cf that assists optimization process
        # training_iteration_vec = np.zeros(shape=(self.num_envs, 1), dtype=np.float32)
        # training_iteration_vec[:] = 10000
        # self.env.set_info({'training_iteration': training_iteration_vec})  # for assisting parameter optimization
        # err_reward_coeff_vec = np.zeros(shape=(self.num_envs, 1), dtype=np.float32)
        # err_reward_coeff_vec[:] = 10.
        # self.env.set_info({'err_reward_coeff': err_reward_coeff_vec})  # for assisting parameter optimization

    # ********************************* Function for bayesian optimization ********************************* #
    def bayesian_optimisation(self, n_iters, x0=None, n_pre_samples=5,
                              gp_params=None, random_search=0, alpha=1e-2, epsilon=1e-7,
                              use_ucb=True):
        """ multi-env bayesian_optimisation

        Uses Gaussian Processes to optimise the loss function `fitness`.

        Arguments:
            n_iters: integer.
                Number of iterations to run the search algorithm.
            x0: array-like, shape = [n_pre_samples, num_envs, n_params].
                Array of initial points to sample the loss function for. If None, randomly
                samples from the loss function.
            n_pre_samples: integer.
                If x0 is None, samples `n_pre_samples` initial points from the loss function.
            gp_params: dictionary.
                Dictionary of parameters to pass on to the underlying Gaussian Process.
            random_search: integer.
                Flag that indicates whether to perform random search or L-BFGS-B optimisation
                over the acquisition function.
            alpha: double.
                Variance of the error term of the GP.
            epsilon: double.
                Precision tolerance for floats.
        """

        x_list = [] # len: n_pre_samples + n_iters; element dim: [ngroups, xdim]
        y_list = [] # len: n_pre_samples + n_iters; element dim: [ngroups, ]

        if x0 is None:
            for i in range(n_pre_samples):
                params = np.random.uniform(self.custom_bound_per_env_lower, self.custom_bound_per_env_upper,
                                           (self.num_groups, self.dim))
                x_list.append(params)
                y_list.append(self.fitness(params).cpu().numpy()) # evaluated value of env_id th env
        else:
            for params in x0:
                x_list.append(params)
                y_list.append(self.fitness(params).cpu().numpy())

        xp = np.array(x_list) # dim: [n_pre_samples, ngroups, xdim]
        yp = np.array(y_list) # dim: [n_pre_samples, ngroups]
        ypmean = np.mean(yp, axis=0) # dim: [ngroups, ]
        ypstd = np.std(yp,  axis=0) # dim: [ngroups, ]

        # optimization iteration for per env
        for n in range(n_iters):
            print("Optimization round ", n)
            ypmean = np.mean(yp, axis=0)  # dim: [ngroups, ]
            ypstd = np.std(yp, axis=0)  # dim: [ngroups, ]
            ypmax = np.max(yp, axis=0) # dim: [ngroups, ]
            ypmin = np.min(yp, axis=0) # dim: [ngroups, ]
            yp = (yp - ypmin) / (ypmax - ypmin) # dim: [n_pre_samples + n_iters, ngroups]

            start = time.time()
            self.advance(xp, yp, ypmin, ypmax)

            # sequential GP
            next_sample_group = np.zeros((self.num_groups, self.dim))  # all groups
            for env_id in range(self.num_groups):
                next_sample = self.get_next_sample(env_id)
                next_sample_group[env_id] = next_sample
                # self.next_sample_all[env_id*self.cfg['group_envs']: (env_id+1)*self.cfg['group_envs']] = next_sample
            print("next_sample ", next_sample)
            print("************ bayes optimize elapse time: ", time.time() - start)

            # Sample loss for new set of parameters/ evaluate the traj with next samples of all env (all envs)
            cv_score_group = self.fitness(next_sample_group) # cv_score_group dim: [ngroups, ], self.next_sample_all dim: [nenvs, xdim]

            # Update lists
            x_list.append(next_sample_group)
            y_list.append(cv_score_group.cpu().numpy())

            # Update xp and yp.
            xp = np.array(x_list) # dim: [n_pre_samples + n_iters, ngroups, xdim]
            yp = np.array(y_list) # dim: [n_pre_samples + n_iters, ngroups]

        #     # if self.cfg['environment']['num_envs'] == 1: # for saving mu and sigma
        #     #     print("----------------- Saving MSO results ... ----------------")
        #     #     # self.mu_history.append(self.concurr.mu)  # save predicted mu in history(Sanity check)
        #     #     # self.sigma_history.append(self.concurr.sigma)  # save predicted sigma in history(Sanity check)
        #     #     self.next_sample_history.append(np.squeeze(self.next_sample_all))
        #     #     self.cv_score_history.append(np.squeeze(-cv_score_group))  # for optimizer logging

        return np.array(x_list), np.array(y_list), ypmean, ypstd


    # ********************************* Function for bayesian optimization ********************************* #
    def advance(self, xp, yp, ypmin, ypmax): # update the value used in get_next_sample
        self.xp = xp
        self.yp = yp
        self.ypmin = ypmin
        self.ypmax = ypmax

    def get_next_sample(self, env_id): # env id for all groups
        train_xp = np.copy(self.xp[:, env_id])  # dim: [n_pre_samples + n_iters, xdim]
        train_yp = np.copy(self.yp[:, env_id])  # dim: [n_pre_samples + n_iters, ]
        model = self.models[env_id]  # use the i-th GP model
        # self.models[env_id].fit(train_xp, train_yp)
        model.fit(train_xp, train_yp)                                     # for sklearn-based guassin process
        # model.updateModel(train_xp, train_yp[:,np.newaxis], None, None) # for GPy-based guassin process

        # if self.num_envs == 1: # for visualizig the guassian process
        #     self.get_mu_sigma(model)

        # Sample next hyperparameter
        if self.random_search:
            x_random = np.random.uniform(self.custom_bound_per_env_lower, self.custom_bound_per_env_upper,
                                         size=(self.random_search, self.dim))

            if self.use_ucb:
                ei = -1 * ucb(x_random, model, n_params=self.dim)
            else:
                ei = -1 * expected_improvement(x_random, model, self.yp, greater_is_better=False, n_params=self.dim)

            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, self.yp, greater_is_better=False,
                                                     bounds=self.custom_bound_per_env, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - self.xp[:, env_id]) <= self.epsilon):
            next_sample = np.random.uniform(self.custom_bound_per_env_lower, self.custom_bound_per_env_upper,
                                            self.custom_bound_per_env.shape[0])
        return next_sample

    def fitness(self, x): # x dim: (ngroups, x_dim). Collect samples for evaluation(all envs) (TORCH-BASED)
        app = np.copy(x)

        # expend the group x to all envs' x
        app_all = np.zeros((self.num_envs, self.dim))  # all envs
        for i in range(self.num_groups):
            app_all[i * self.cfg['group_envs']: (i + 1) * self.cfg['group_envs']] = app[i]

        # set mu info before reset. No concatenate. no change in actuator model
        # app = np.reshape(app, (self.num_envs, -1))  # app dim: (env_num, x_dim)
        # self.env.set_info({'set_obs_UP': app})
        # TODO: pass and set observation in envs
        self.env.obs_dict['UP'] = torch.from_numpy(app_all).to(self.device).to(torch.float)

        avg_perf = []
        perf_vector = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        perf_vector_group = torch.zeros(self.num_groups, dtype=torch.float, device=self.device, requires_grad=False)

        t = time.time()
        for _ in range(self.eval_num):
            if self.test_mode:
                self.scenario.reset()  # need to refresh scenario when starting a new traj. The pitch start pose is set by calling env reset
                self.test_steps = self.scenario.test_step
                # obs = self.scenario.ob_init
                # info = self.scenario.env.get_info()
            # else:
            #     obs, _ = self.env.reset()
            # _, _ = self.env.reset()
            obs_dict = self.env.get_observations()  # don not need to reset every time, just reset when reach termination
            obs, UP = obs_dict['obs'].to(self.device), obs_dict['UP'].to(self.device)
            obs_UP = torch.cat([obs, UP], dim=-1)

            rollout_rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(self.test_steps):
                if self.test_mode:
                    _ = self.scenario.advance()

                actions = self.policy(obs_UP.detach()) # critic obs same as obs
                # TODO: how to fix the env / create a new env for the optimizer??
                obs_dict, rewards, dones, infos = self.env.step(actions.detach()) # rewards dim: (num_envs, )
                obs, UP = obs_dict['obs'].to(self.device), obs_dict['UP'].to(self.device)
                obs_UP = torch.cat([obs, UP], dim=-1)

                perf_vector += rewards # add rewards at current step
                rollout_rew += rewards # rollout rewards for one evaluation
            avg_perf.append(rollout_rew)

        # print("avg_perf", -np.mean(np.array(avg_perf)))
        print("perf_vector", -perf_vector)
        print("************ evaluation for once elapse time: ", time.time() - t)

        for i in range(self.num_groups):
            perf_vector_group[i] = torch.sum(perf_vector[self.cfg['group_envs']*i : self.cfg['group_envs']*(i+1)]) / (self.eval_num * self.cfg['group_envs'])

        # For sanity check and Optimization(save solution hist)
        if self.test_mode:
            self.save_best_callback(app, perf_vector)

        return -perf_vector_group # return performance vector containing preformance of all group envs. dim: (n_groups, )

    def save_best_callback(self, app, perf_vector): # dim: app(env_num, x_dim); performance vector(n_groups, )
        if -perf_vector[0] < self.best_f:
            self.best_x = np.copy(app[0])
            self.best_f = -perf_vector[0].cpu().numpy()
        print("self.best_x: ", self.best_x)
        print("self.best_f: ", self.best_f)
        temp_best_x = np.copy(self.best_x)
        temp_best_f = np.copy(self.best_f)
        self.solution_history.append(np.squeeze(temp_best_x))
        self.best_fitness_history.append(np.squeeze(-temp_best_f))

    def optimize(self, maxiter = 20, x0=None):
        # dim: xs: [n_pre_samples + n_iters, ngroups, xdim]. ys: [n_pre_samples + n_iters, ngroups]
        xs, ys, _, _ = self.bayesian_optimisation(maxiter, x0=x0, random_search = 1000)
        min_idx = np.argmin(ys, axis=0) # dim: [ngroups, ]
        xopt= np.array([xs[:,i,:][min_idx[i]] for i in range(min_idx.shape[0])]) # dim: [ngroups, xdim]
        # print('optimized: ', repr(xopt))

        # augment to all envs
        xopt_all = np.zeros((self.num_envs, self.dim))
        for i in range(self.num_groups):
            xopt_all[i*self.cfg['group_envs']: (i+1)*self.cfg['group_envs']] = xopt[i]
        self.best_x = xopt_all

        return xopt_all, xopt # xopt_all dim: [nenvs, xdim]; xopt: [ngroups, xdim]


# class ConcurrVar:
#     def __init__(self, cfg, dim, custom_bound_per_env):
#         self.cfg = cfg
#         self.num_envs = self.cfg['environment']['num_envs']
#         self.num_groups = int(self.cfg['environment']['num_envs']/ self.cfg['environment']['group_envs'])
#         self.dim = dim
#         self.custom_bound_per_env = custom_bound_per_env
#         self.random_search = 1000
#         self.use_ucb = True
#         self.epsilon = 1e-7
#         self.alpha = 1e-2
#
#         # Create the GP using sklearn
#         kernel = gp.kernels.Matern(length_scale_bounds=(1e-2, 1e3)) + gp.kernels.WhiteKernel()
#         model = gp.GaussianProcessRegressor(kernel=kernel,
#                                             alpha=self.alpha,
#                                             n_restarts_optimizer=10,
#                                             normalize_y=True)
#
#         # # Create GP using Gyopt
#         # from .gpmodel.gpmodel import GPModel
#         # model = GPModel(optimize_restarts=10) # use the default kernel
#
#         self.models = [model] * self.num_envs
#
#     def advance(self, xp, yp, ypmin, ypmax): # update the value used in get_next_sample
#         self.next_sample_all = np.zeros((self.num_envs, self.dim))  # all envs
#         self.next_sample_group = np.zeros((self.num_groups, self.dim))  # all groups
#         self.xp = xp
#         self.yp = yp
#         self.ypmin = ypmin
#         self.ypmax = ypmax
#
#     def get_mu_sigma(self, model):
#         # ********************************************************************************************* #
#         # ************************** For sanity check plot(Use sequential GP)************************** #
#         # ********************************************************************************************* #
#         boom_grid = np.linspace(0.8, 1.2, 20)
#         dipper_grid = np.linspace(0.8, 1.2, 20)
#         tele_grid = np.linspace(0.3, 1.2, 80)
#         pitch_grid = np.linspace(0.8, 1.2, 20)
#         boom_set, dipper_set, tele_set, pitch_set = np.meshgrid(boom_grid, dipper_grid, tele_grid, pitch_grid)
#
#         mu, sigma = model.predict(np.c_[boom_set.ravel(), dipper_set.ravel(), tele_set.ravel(), pitch_set.ravel()], return_std=True)  # predict mean and var of guassian process. mu: [ngrids, ], sigma[ngrids, ]
#         mu = -1 * (mu * (self.ypmax - self.ypmin) + self.ypmin)  # un-normalize the prediction
#         print("mu: ", mu)
#
#         # reshape it into (num_sample, num_sample, num_sample, num_sample)
#         self.mu, self.sigma = mu.reshape(boom_set.shape), sigma.reshape(boom_set.shape)
#         # return mu, sigma
#         # ********************************************************************************************* #
#         # ************************** For sanity check plot(Use sequential GP)************************** #
#         # ********************************************************************************************* #
#
#
#     def get_next_sample(self, env_id): # env id for all groups
#         n_params = self.custom_bound_per_env.shape[0]
#         train_xp = np.copy(self.xp[:, env_id])  # dim: [n_pre_samples + n_iters, 4]
#         train_yp = np.copy(self.yp[:, env_id])  # dim: [n_pre_samples + n_iters, ]
#         model = self.models[env_id]  # use the i-th GP model
#         # self.models[env_id].fit(train_xp, train_yp)
#         model.fit(train_xp, train_yp)                                     # for sklearn-based guassin process
#         # model.updateModel(train_xp, train_yp[:,np.newaxis], None, None) # for GPy-based guassin process
#
#         # if self.num_envs == 1: # for visualizig the guassian process
#         #     self.get_mu_sigma(model)
#
#         # Sample next hyperparameter
#         if self.random_search:
#             x_random = np.random.uniform(self.custom_bound_per_env[:, 0], self.custom_bound_per_env[:, 1],
#                                          size=(self.random_search, n_params))
#
#             if self.use_ucb:
#                 ei = -1 * ucb(x_random, model, n_params=n_params)
#             else:
#                 ei = -1 * expected_improvement(x_random, model, self.yp, greater_is_better=False, n_params=n_params)
#
#             next_sample = x_random[np.argmax(ei), :]
#         else:
#             next_sample = sample_next_hyperparameter(expected_improvement, model, self.yp, greater_is_better=False,
#                                                      bounds=self.custom_bound_per_env, n_restarts=100)
#
#         # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
#         if np.any(np.abs(next_sample - self.xp[:, env_id]) <= self.epsilon):
#             next_sample = np.random.uniform(self.custom_bound_per_env[:, 0], self.custom_bound_per_env[:, 1],
#                                             self.custom_bound_per_env.shape[0])
#
#         return next_sample
