import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor, ResultsWriter

from drone_area_coverage import DroneEnvironment



def run_agent(thread_id, lock, map_size, drone_num, view_range, policy, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, policy_kwargs, verbose, learning_steps, exp_name):
    # Create models directory
    model_dir = exp_name + '/models/'
    os.makedirs(model_dir, exist_ok=True)
        
    # Create log directory
    model_name = 'agent' + str(thread_id)
    log_dir = exp_name + '/tmp/agent' + str(thread_id) + '/'
    os.makedirs(log_dir, exist_ok=True)
    
    start_time = time.perf_counter()
    print(f"Thread {thread_id} started at time {start_time}")
    
    env = DroneEnvironment(thread_id, lock, map_size, drone_num, view_range, exp_name)
    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(log_dir=log_dir, model_dir=model_dir, model_name=model_name)
    model = PPO(policy=policy, env=env, tensorboard_log=exp_name, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef,  policy_kwargs=policy_kwargs, verbose=verbose)
    model.learn(total_timesteps=learning_steps, callback=callback)
    print(f"Thread {thread_id} try to finish...")
    env.finish()
    
    finish_time = time.perf_counter()
    duration = round((finish_time - start_time)/60)
    print(f"Thread {thread_id} finished at time {finish_time} in {duration} minutes!")
    

    # Save model
    path_model = exp_name + '\\models\\agent' + str(thread_id)
    model.save(path_model)
    
    
def run_agent_parallel(thread_id, locks, map_size, drone_num, view_range, policy, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, policy_kwargs, verbose, learning_steps, exp_name, env_names):
    # Function used for parallelizing environments
    def make_env(thread_id, lock, map_size, drone_num, view_range, exp_env_name):
        def _init():
            return DroneEnvironment(thread_id, lock, map_size, drone_num, view_range, exp_env_name)
        return _init

    # Prepare environments in VecEnc format
    envs_list = []
    for i in range(len(env_names)):
        exp_env_name = exp_name + '/' + env_names[i]
        envs_list.append(make_env(thread_id=thread_id, lock=locks[i], map_size=map_size, drone_num=drone_num, 
                                  view_range=view_range, exp_env_name=exp_env_name))
    vec_envs = DummyVecEnv(envs_list)
    
    
    # Create models directory
    model_dir = exp_name + '/models/'
    os.makedirs(model_dir, exist_ok=True)
    
    # Create log directory
    model_name = 'agent' + str(thread_id)
    log_dir = exp_name + '/tmp/agent' + str(thread_id) + '/'
    os.makedirs(log_dir, exist_ok=True)
  
    # Wrap the VecEnv
    vec_envs = VecMonitor(vec_envs, filename=str(log_dir + 'monitor.csv')) 
    
    # Start learning in parallel
    start_time = time.perf_counter()
    print(f"Thread {thread_id} started at time {start_time}")
    
    callback = SaveOnBestTrainingRewardCallback(log_dir=log_dir, model_dir=model_dir, model_name=model_name)
    model = PPO(policy=policy, env=vec_envs, tensorboard_log=exp_name, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef,  policy_kwargs=policy_kwargs, verbose=verbose)
    model.learn(total_timesteps=learning_steps, callback=callback)
    
    # Finishing learning
    print(f"Thread {thread_id} try to finish...")
    vec_envs.env_method('finish')
    vec_envs.close()    

    finish_time = time.perf_counter()
    duration = round((finish_time - start_time)/60)
    print(f"Thread {thread_id} finished at time {finish_time} in {duration} minutes!")
    
    # Save model
    path_model = exp_name + '\\models\\agent' + str(thread_id)
    model.save(path_model)


def continue_agent(thread_id, lock, map_size, drone_num, view_range, custom_objects, learning_steps, exp_name, model_path):
    # Create models directory
    model_dir = exp_name + '/models/'
    os.makedirs(model_dir, exist_ok=True)
        
    # Create log directory
    model_name = 'agent' + str(thread_id)
    log_dir = exp_name + '/tmp/'
    os.makedirs(log_dir, exist_ok=True)
    
    start_time = time.perf_counter()
    print(f"Thread {thread_id} started at time {start_time}")
    
    env = DroneEnvironment(thread_id, lock, map_size, drone_num, view_range, exp_name)
    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(log_dir=log_dir, model_dir=model_dir, model_name=model_name)
    model = PPO.load(model_path, custom_objects=custom_objects, env=env)
    model.learn(total_timesteps=learning_steps, callback=callback)
    print(f"Thread {thread_id} try to finish...")
    env.finish()
    
    finish_time = time.perf_counter()
    duration = round((finish_time - start_time)/60)
    print(f"Thread {thread_id} finished at time {finish_time} in {duration} minutes!")
    

    # Save model
    path_model = exp_name + '\\models\\agent' + str(thread_id)
    model.save(path_model)
    

def continue_agent_parallel(thread_id, locks, map_size, drone_num, view_range, custom_objects, learning_steps, exp_name, env_names, model_path):
    # Function used for parallelizing environments
    def make_env(thread_id, lock, map_size, drone_num, view_range, exp_env_name):
        def _init():
            return DroneEnvironment(thread_id, lock, map_size, drone_num, view_range, exp_env_name)
        return _init

    # Prepare environments in VecEnc format
    envs_list = []
    for i in range(len(env_names)):
        exp_env_name = exp_name + '/' + env_names[i]
        envs_list.append(make_env(thread_id=thread_id, lock=locks[i], map_size=map_size, drone_num=drone_num, 
                                  view_range=view_range, exp_env_name=exp_env_name))
    vec_envs = SubprocVecEnv(envs_list, start_method='spawn')    
    
    # Create models directory
    model_dir = exp_name + '/models/'
    os.makedirs(model_dir, exist_ok=True)
    
    # Create log directory
    model_name = 'agent' + str(thread_id)
    log_dir = exp_name + '/tmp/'
    os.makedirs(log_dir, exist_ok=True)
  
    # Wrap the VecEnv
    vec_envs = VecMonitor(vec_envs, filename=str(log_dir + 'monitor.csv')) 
    
    # Start learning in parallel
    start_time = time.perf_counter()
    print(f"Thread {thread_id} started at time {start_time}")
    
    callback = SaveOnBestTrainingRewardCallback(log_dir=log_dir, model_dir=model_dir, model_name=model_name)
    model = PPO.load(model_path, custom_objects=custom_objects, env=vec_envs)
    model.learn(total_timesteps=learning_steps, callback=callback)
    
    # Finishing learning
    print(f"Thread {thread_id} try to finish...")
    vec_envs.env_method('finish')
    vec_envs.close()
    
    print(f"Thread {thread_id} finished his learning.")
    

    finish_time = time.perf_counter()
    duration = round((finish_time - start_time)/60)
    print(f"Thread {thread_id} finished at time {finish_time} in {duration} minutes!")
    
    # Save model
    path_model = exp_name + '\\models\\agent' + str(thread_id)
    model.save(path_model)    


def test_agent(thread_id, lock, map_size, drone_num, view_range, exp_name):
    #path_model = exp_name + '\\models\\agent' + str(thread_id)
    path_model = exp_name + '\\models\\best_agent' + str(thread_id)
    model = PPO.load(path_model)
    
    env = DroneEnvironment(thread_id, lock, map_size, drone_num, view_range, exp_name)
    
    total_reward = 0
    obs = env.reset()
    while True:   
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Thread {thread_id} finished with reward {total_reward}")
    

def test_agent_mean(thread_id, lock, map_size, drone_num, view_range, exp_name, episodes):
    #path_model = exp_name + '\\models\\agent' + str(thread_id)
    path_model = exp_name + '\\models\\best_agent' + str(thread_id)
    model = PPO.load(path_model)
    
    env = DroneEnvironment(thread_id, lock, map_size, drone_num, view_range, exp_name)
    
    total_reward = 0
    for i in range(episodes):
        ep_reward = 0
        obs = env.reset()
        while True:   
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                total_reward += ep_reward
                break
    mean_reward = total_reward / episodes
    
    print(f"Thread {thread_id} finished with mean reward {mean_reward}")
    

def random_agent(thread_id, lock, map_size, drone_num, view_range, exp_name):  
    env = DroneEnvironment(thread_id, lock, map_size, drone_num, view_range, exp_name)
    
    total_reward = 0
    obs = env.reset()
    while True:   
        action = np.random.randint(0, env.action_space.n - 1)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Thread {thread_id} finished with reward {total_reward}")

    
    
    
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, log_dir: str, model_dir: str, model_name: str='agent', check_freq: int=1500):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose=0)
        
        best_model_name = str('best_' + model_name)
        last_model_name = str('last_' + model_name)
        
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.save_path = os.path.join(model_dir, best_model_name)
        self.save_path_last = os.path.join(model_dir, last_model_name)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.model_dir is not None:
            os.makedirs(self.model_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                
            self.model.save(self.save_path_last)
        return True

    
class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename = None, info_keywords = ()):
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        VecEnvWrapper.__init__(self, venv)
        self.episode_returns = None
        self.episode_lengths = None
        self.episode_count = 0
        self.t_start = time.time()

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        if filename:
            self.results_writer = ResultsWriter(
                filename, header={"t_start": self.t_start, "env_id": env_id}, extra_keys=info_keywords
            )
        else:
            self.results_writer = None
        self.info_keywords = info_keywords

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos
    
    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()