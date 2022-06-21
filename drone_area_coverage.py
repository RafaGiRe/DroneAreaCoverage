import numpy as np
import random
import gym
import time
import os



class MapLegend(object):
    def __init__(self):
        # MAP LEGEND
        self.CELL_EMPTY = 0
        self.CELL_TARGET_UNOBS = 1
        self.CELL_TARGET_OBS = 2
        self.CELL_FORBIDDEN = 3
        
        # OBSTACLES MAP LEGEND
        self.FORBIDDEN = -1
        self.EMPTY = 1
        
        # OBSERVED MAP LEGEND
        self.TARGET_OBS = -1
        self.TARGET_UNOBS = 1
        
        # DRON MAP LEGEND
        self.NO_DRON = -1
        self.CURRENT_DRON = 1
        self.OTHER_DRON = 1
        

class RewardValues(object):
    def __init__(self):
        # DISCOVERING
        self.UNOBSERVED_TARGET_CELL = 3
#         self.UNOBSERVED_TARGET_CELL_DELTA = 20    # REWARD SHAPING
        
        # FORBIDDEN CELL
        self.FORBIDDEN_CELL = -100
        # OUT OF MAP
        self.OUT_CELL = -100
        # COLLISION
        self.COLLISION = -100
        
        # FINISH
        self.DONE = 100
        self.NOT_DONE = -100
        self.REMAINING_TARGET_CELL = 0 # -3
        
        # ENERGY
#         self.EN_0 = -0.5 # WAITING COST
#         self.EN_1 = -1 # STRAIGHT COST
#         self.EN_2 = -1.5 # 45º COST
#         self.EN_3 = -2 # 90º COST
#         self.EN_4 = -2.5 # 135º COST
#         self.EN_5 = -3 # 180º COST
#         # ROWS = LAST_ACTION, COLUMNS = NEW_ACTION
#         self.ENERGY_MATRIX = np.array(
#             [
#             [self.EN_1, self.EN_2, self.EN_3, self.EN_2, self.EN_0, self.EN_4, self.EN_3, self.EN_4, self.EN_5],
#             [self.EN_2, self.EN_1, self.EN_2, self.EN_3, self.EN_0, self.EN_3, self.EN_4, self.EN_5, self.EN_4],
#             [self.EN_3, self.EN_2, self.EN_1, self.EN_4, self.EN_0, self.EN_2, self.EN_5, self.EN_4, self.EN_3],
#             [self.EN_2, self.EN_3, self.EN_4, self.EN_1, self.EN_0, self.EN_5, self.EN_2, self.EN_3, self.EN_4],
#             [self.EN_1, self.EN_1, self.EN_1, self.EN_1, self.EN_0, self.EN_1, self.EN_1, self.EN_1, self.EN_1],
#             [self.EN_4, self.EN_3, self.EN_2, self.EN_5, self.EN_0, self.EN_1, self.EN_4, self.EN_3, self.EN_2],
#             [self.EN_3, self.EN_4, self.EN_5, self.EN_2, self.EN_0, self.EN_4, self.EN_1, self.EN_2, self.EN_3],
#             [self.EN_4, self.EN_5, self.EN_4, self.EN_3, self.EN_0, self.EN_3, self.EN_2, self.EN_1, self.EN_2],
#             [self.EN_5, self.EN_4, self.EN_3, self.EN_4, self.EN_0, self.EN_2, self.EN_3, self.EN_2, self.EN_1],
#             ])
        
        # SIMPLIFIED VERSION
        self.EN_0 = -5 # WAITING COST
        self.EN_1 = -1 # STRAIGHT COST
        self.EN_2 = -1 # 45º COST
        self.EN_3 = -1 # 90º COST
        self.EN_4 = -1 # 135º COST
        self.EN_5 = -1 # 180º COST
        # ROWS = LAST_ACTION, COLUMNS = NEW_ACTION
        self.ENERGY_MATRIX = np.array(
            [
            [self.EN_0, self.EN_1, self.EN_3, self.EN_3, self.EN_5],
            [self.EN_0, self.EN_1, self.EN_3, self.EN_3, self.EN_5],
            [self.EN_0, self.EN_3, self.EN_1, self.EN_5, self.EN_3],
            [self.EN_0, self.EN_3, self.EN_5, self.EN_1, self.EN_3],
            [self.EN_0, self.EN_5, self.EN_3, self.EN_3, self.EN_1],
            ])

        
        
class Dron(object):
    def __init__(self, identifier, map_identifier, pos, view_range):
        self.id = identifier
        self.map_id = map_identifier
        self.pos = pos
        self.view_range = view_range

        
class DroneEnvironment(gym.Env):
    metadata = {'render.modes': ['human'], 'name': 'drone_area_coverage'}
    
    def __init__(self, current_agent_id, lock, map_size, drone_num, view_range, exp_name):
        super(DroneEnvironment, self).__init__()
        
        self.exp_name = exp_name
        self.env_file = exp_name + '\\shared_environment.npz'
        self.semaphore_file = exp_name + '\\semaphore.npy'
        self.lock = lock
        self.timestep = 0.05
        self.max_tries = 50
        self.episode = 0
        self.steps = 0
        
        if(drone_num == 1):
            obs_space_shape = (3, map_size[0], map_size[1])
        else:
            obs_space_shape = (4, map_size[0], map_size[1])     # MULTIAGENT
        act_space_shape = 5
        
        
        self.map_size = map_size
        self.drone_num = drone_num
        self.view_range = view_range
        self.max_steps = 150
        
        # AGENT 0 IS THE MANAGER OF THE ENVIRONMENT (reset)
        self.id = current_agent_id
        self.main_thread = (self.id == 0)
        if (self.main_thread): self._create_dir(exp_name)
        self._initialize_semaphore()
        self.turn = False
        
        self.map_legend = MapLegend()
        self.reward_values = RewardValues()

        self.map = np.ones(self.map_size, dtype=np.int)*self.map_legend.CELL_EMPTY
        self.obstacles_map = np.ones(self.map_size, dtype=np.int)*self.map_legend.EMPTY
        self.observed_map = np.ones(self.map_size, dtype=np.int)*self.map_legend.TARGET_UNOBS
        self.drone_map = np.ones(self.map_size, dtype=np.int)*self.map_legend.NO_DRON
        self.other_dron_map = np.ones(self.map_size, dtype=np.int)*self.map_legend.NO_DRON
        self.current_dron_map = np.ones(self.map_size, dtype=np.int)*self.map_legend.NO_DRON
        
        self.agents = {}
        self.rewards = {}
        self.actions = {}
        self.dones = {}
        self.resets = {}
        self.actives = {k: True for k in range(self.drone_num)}
        
        self.state = np.empty(obs_space_shape, dtype=np.int)
        self.shared_state = np.empty((3, map_size[0], map_size[1]), dtype=np.int)

        
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=obs_space_shape, dtype=np.int)
        self.action_space = gym.spaces.Discrete(act_space_shape)

        
        
    ################################ SEMAPHORE METHODS ################################
    def _create_dir(self, exp_name):
        # Create the dirs
        os.makedirs(exp_name, exist_ok=True)
        return
    
    def _initialize_semaphore(self):
        semaphore = {'turn': 0, 'main_thread': 0, 'reset_completed': False}
        self.semaphore = semaphore
        
        if(self.main_thread): 
            np.save(self.semaphore_file, self.semaphore)
        
        return
     
    def _acquire(self):
        turn = False
        count = 0
        while not turn:
            count += 1
            
            self.lock.acquire()
            self.semaphore = np.load(self.semaphore_file, allow_pickle=True).item()
            agent_turn = self.semaphore['turn']
            if (self.semaphore['turn'] == self.id):
                turn = True
                
                if (self.semaphore['main_thread'] == self.id):
                    self.main_thread = True
                
                break
            self.lock.release()
            
            if count > self.max_tries:
                break
            time.sleep(self.timestep)
            
        self.turn = turn
        return
    
    def _release(self):
        for i in range(1, self.drone_num + 1):
            agent = (self.id + i) % self.drone_num
            if (self.actives[agent]):
                self.semaphore['turn'] = agent
                break
        
        np.save(self.semaphore_file, self.semaphore)
        self.lock.release()
        self.turn = False
        
        return True

    def _ready_to_start(self):
        self.semaphore = np.load(self.semaphore_file, allow_pickle=True).item()
        return self.semaphore['reset_completed']
    
    def _unmark_reset(self):
        self.semaphore = np.load(self.semaphore_file, allow_pickle=True).item()
        self.semaphore['reset_completed'] = False
        np.save(self.semaphore_file, self.semaphore)
        return True
    
    def finish(self):
        if not (self.turn):
            self._acquire()
        
        if self.turn:
            self._load_environment()
            self._update_state()
            
            # First, mark itself as deactivated
            self.actives[self.id] = False
            
            # Now, remove this drone from the map
            self.drone_map[np.where(self.drone_map == self.agents[self.id].map_id)] = 0
            self.shared_state[0] = self.obstacles_map
            self.shared_state[1] = self.observed_map
            self.shared_state[2] = self.drone_map
            
            self._save_environment()
            
            
            # If main thread, we should mark another as the main
            if self.main_thread:
                # Now, check if there is still an active agent for marking as main_thread
                for i in range(1, self.drone_num + 1):
                    candidate = (self.id + i) % self.drone_num
                    if (self.actives[candidate]):
                        self.semaphore['main_thread'] = candidate
                        #self.semaphore['turn'] = candidate
                        break
            self._release()
        else:
            # ERROR?
            print(f'Thread {self.id} fails to get the turn...')
            pass
        return
    
    
    
    ################################ SAVE/LOAD ENV METHODS ################################
    def _save_environment(self):
        np.savez(self.env_file, map=self.map, state=self.shared_state, agents=self.agents, actions=self.actions, rewards=self.rewards, dones=self.dones, resets=self.resets, actives=self.actives)

    def _load_environment(self):
        env_data = np.load(self.env_file, allow_pickle=True)
        self.map = env_data['map']
        self.shared_state =  env_data['state']
        self.agents = env_data['agents'].item()
        self.actions = env_data['actions'].item()
        self.rewards = env_data['rewards'].item()
        self.dones = env_data['dones'].item()
        self.resets = env_data['resets'].item()
        self.actives = env_data['actives'].item()
        
    def _update_state(self):
        self.obstacles_map = self.shared_state[0]
        self.observed_map = self.shared_state[1]
        self.drone_map = self.shared_state[2]
        
        self.current_dron_map = np.ones(self.map_size, dtype=np.int) * self.map_legend.NO_DRON
        self.current_dron_map[np.where(self.drone_map == self.agents[self.id].map_id)] = self.map_legend.CURRENT_DRON
        
        self.other_dron_map = np.ones(self.map_size, dtype=np.int) * self.map_legend.NO_DRON
        self.other_dron_map[np.where(self.drone_map != 0)] = self.map_legend.OTHER_DRON
        self.other_dron_map[np.where(self.drone_map == self.agents[self.id].map_id)] = self.map_legend.NO_DRON
        
        
        self.state[0] = self.obstacles_map
        self.state[1] = self.observed_map
        self.state[2] = self.current_dron_map
        if(self.drone_num > 1):
            self.state[3] = self.other_dron_map     # MULTIAGENT

        
        return
    
    
    
    ################################ RESET METHODS ################################
    def _reset_environment(self):  
        self._reset_map()
        self._reset_obstacles_map()
        self._reset_drones()
        self._reset_drone_map()
        self._reset_observed_map()
        
        self.shared_state = np.empty((3, self.map_size[0], self.map_size[1]), dtype=np.int)
        self.shared_state[0] = self.obstacles_map
        self.shared_state[1] = self.observed_map
        self.shared_state[2] = self.drone_map
        
        self._save_environment()
        
        self.semaphore['turn'] = self.semaphore['main_thread']
        self.semaphore['reset_completed'] = True
        
        np.save(self.semaphore_file, self.semaphore)
        

        return self.shared_state

    def _reset_map(self):
        self.map = self._generate_random_map()
        return
    
    def _generate_random_map(self):
#         little_map = True
        
#         # Notation for each shape: (height, width)
        
#         # Map will contain three Forbidden blocks, which shape is predefined. For them, we will randomly pose its left corner on the map
#         if little_map:
#             forbidden_shapes = np.array([[25, 5], [9, 6], [9, 12]], dtype=int)
#         else:
#             forbidden_shapes = np.array([[15, 28], [26, 10], [25, 22]], dtype=int)
#         forbidden_positions = np.empty(forbidden_shapes.shape, dtype=int)
#         for i in range(len(forbidden_shapes)):
#             forbidden_positions[i][0] = random.randint(0, self.map_size[0] - 1)
#             forbidden_positions[i][1] = random.randint(0, self.map_size[1] - 1)
        
        
#         # Map will contain 4 different target areas, which shape is predefined. For them, we will randomly pose its left corner on the map
#         if little_map:
#             target_shapes = np.array([[15, 20], [12, 13], [20, 9]], dtype=int)
#         else:
#             target_shapes = np.array([[26, 26], [24, 11], [9, 17], [14, 13]], dtype=int)
#         target_positions = np.empty(target_shapes.shape, dtype=int)
#         for i in range(len(target_shapes)):
#             target_positions[i][0] = random.randint(0, self.map_size[0] - 1)
#             target_positions[i][1] = random.randint(0, self.map_size[1] - 1)
        
        
#         # GENERATE THE MAP
#         # Keep in mind that the map is really vertically inverted (top row is 0, bottom row is map_size.shape[0])
#         # Going up is equivalent to decrease the row, and going down is equivalent to increase the row
        
#         random_map = np.ones(self.map_size, dtype=np.int)*self.map_legend.CELL_EMPTY
        
#         # Pose forbidden shapes
#         for k in range(len(forbidden_shapes)):
#             for i in range(forbidden_shapes[k][0]):
#                 for j in range(forbidden_shapes[k][1]):
#                     x = forbidden_positions[k][0]-i   # Substraction because map vertically inverted
#                     y = forbidden_positions[k][1]+j
                    
#                     if (x<0 or x>=self.map_size[0] or y<0 or y>=self.map_size[1]):
#                         # Out of map
#                         continue
                    
#                     random_map[x, y] = self.map_legend.CELL_FORBIDDEN
        
#         # Pose target shapes
#         for k in range(len(target_shapes)):
#             for i in range(target_shapes[k][0]):
#                 for j in range(target_shapes[k][1]):
#                     x = target_positions[k][0]-i   # Substraction because map vertically inverted
#                     y = target_positions[k][1]+j
                    
#                     if (x<0 or x>=self.map_size[0] or y<0 or y>=self.map_size[1]):
#                         # Out of map
#                         continue
#                     elif (random_map[x, y] == self.map_legend.CELL_FORBIDDEN):
#                         continue
                    
#                     random_map[x, y] = self.map_legend.CELL_TARGET_UNOBS


        # SIMPLIFIED VERSION, ALL TARGET CELLS EXCEPT EDGES
        random_map = np.ones(self.map_size, dtype=np.int)*self.map_legend.CELL_TARGET_UNOBS
        for i in range(self.map_size[0]):
            random_map[i][0] = self.map_legend.CELL_FORBIDDEN
            random_map[i][self.map_size[1] - 1] = self.map_legend.CELL_FORBIDDEN
        for j in range(self.map_size[1]):
            random_map[0][j] = self.map_legend.CELL_FORBIDDEN
            random_map[self.map_size[0] - 1][j] = self.map_legend.CELL_FORBIDDEN

        return random_map
     
        
    def _reset_obstacles_map(self):
        self.obstacles_map = np.ones(self.map_size, dtype=np.int) * self.map_legend.EMPTY
        self.obstacles_map[np.where(self.map == self.map_legend.CELL_FORBIDDEN)] = self.map_legend.FORBIDDEN

        return
        
    def _reset_observed_map(self):        
        # Initially fully unobserved, except the forbidden cells
        self.observed_map = np.ones(self.map_size, dtype=np.int) * self.map_legend.TARGET_OBS
        self.observed_map[np.where(self.map == self.map_legend.CELL_TARGET_UNOBS)] = self.map_legend.TARGET_UNOBS
        
        # Now, we update the Observed map according to the initial position (view) of each drone
        for k in self.agents:
            self._update_observed_map(self.agents[k])
        
        return
        
             
    def _reset_drones(self):       
        multiple = int(255/self.drone_num)
        self.start_pos = (0, 0)    
        self.agents = {i: Dron(i, (i+1)*multiple, self.start_pos, self.view_range) for i in range(self.drone_num)}
        
        drone_positions = []
        for k in self.agents:
            new_pos = None
            
            # Avoid new_pos is a Forbidden cell
            while True:
                new_pos = random.randint(0, self.map_size[0] - 1), random.randint(0, self.map_size[1] - 1)
                if (new_pos not in drone_positions) and (self.map[new_pos] != self.map_legend.CELL_FORBIDDEN):
                    drone_positions.append(new_pos)
                    break
            
            self.agents[k].pos = new_pos
        
        # Reset other attributes
        self.rewards = {self.agents[k].id: 0.0 for k in self.agents}
        self.actions = {self.agents[k].id: 0 for k in self.agents}    # WAIT as the first action
        self.dones = {self.agents[k].id: False for k in self.agents}
        self.resets = {self.agents[k].id: False for k in self.agents}
        # self.actives must not be updated, it marks if an agent is still learning or it has finished. This has been initialized in the init
           
    def _reset_drone_map(self):
        self.drone_map = np.zeros(self.map_size, dtype=np.int)
        for k in self.agents:
            if(self.actives[k]):
                self.drone_map[self.agents[k].pos] = self.agents[k].map_id
        return

    def reset(self):
        self.episode += 1
        self.steps = 0
        
        if not (self.turn):
            self._acquire()
        if self.turn:
            if self.main_thread:
                print(f'Thread {self.id} reset the environment - episode {self.episode}')
                self._reset_environment()
            else:
                ready_to_start = self._ready_to_start()
                while not ready_to_start:
                    if (self.turn):
                        self._release()
                    if not (self.turn):
                        self._acquire()
                    ready_to_start = self._ready_to_start()
                self._load_environment()   
            
            self._update_state()
            self.resets[self.id] = True
            self._save_environment()
            if (self.turn):
                self._release()
        else:
            # ERROR?
            print(f'Thread {self.id} fails to get the turn...')
            pass
        
        
        # To avoid lock probems, we have to take the turn again before starting the first step
        if not (self.turn):
            self._acquire()
        if self.turn:
            self._load_environment()
            self._update_state()
            pass
        else:
            # ERROR?
            print(f'Thread {self.id} fails to get the turn...')
            pass
        
        
        return self.state
    
    
    ################################ STEP METHODS ################################
    def _task_completed(self):
        completed = True
        
        for i in range(self.map_size[0]):
            if not completed:
                break
            for j in range(self.map_size[1]):
                if self.map[i, j] == self.map_legend.CELL_TARGET_UNOBS:
                    completed = False
                    break
        return completed
    
    def _task_failed(self):
        failed = False
        
        if(self.map[self.agents[self.id].pos] == self.map_legend.CELL_FORBIDDEN):
            failed = True
        
        for k in self.agents:
                if(self.agents[k].pos==self.agents[self.id].pos and k != self.id and (self.actives[k])):
                    failed = True
                    break
        
        return failed
    
    def _drone_step(self, drone, action):
        
        # Keep in mind that the map is really vertically inverted (top row is 0, bottom row is map_size.shape[0])
        # Going up is equivalent to decrease the row, and going down is equivalent to increase the row
        
        ind_reward = 0
        
        pos_reward = 0
        obs_reward = 0
        collision_reward = 0
        energy_reward = 0
        
        new_pos = None
        action_aborted = False
        
        action = int(action)
        
        
        
        # Legend:
            # Drone is centered (position 4)
            # [0] [1] [2]
            # [3] [4] [5]
            # [6] [7] [8]
#         if action == 0:
#             # Move to North West (NW)
#             new_pos = (drone.pos[0] - 1, drone.pos[1] - 1)
#         elif action == 1:
#             # Move to N
#             new_pos = (drone.pos[0] - 1, drone.pos[1])
#         elif action == 2:
#             # Move to NE
#             new_pos = (drone.pos[0] - 1, drone.pos[1] + 1)
#         elif action == 3:
#             # Move to W
#             new_pos = (drone.pos[0], drone.pos[1] - 1)
#         elif action == 4:
#             # WAIT
#             new_pos = (drone.pos[0], drone.pos[1])
#         elif action == 5:
#             # Move to E
#             new_pos = (drone.pos[0], drone.pos[1] + 1)
#         elif action == 6:
#             # Move to SW
#             new_pos = (drone.pos[0] + 1, drone.pos[1] - 1)
#         elif action == 7:
#             # Move to S
#             new_pos = (drone.pos[0] + 1, drone.pos[1])
#         elif action == 8:
#             # Move to SE
#             new_pos = (drone.pos[0] + 1, drone.pos[1] + 1)
#         else:
#             # UNKNOWN ACTION - ERROR
#             pass
        
    
        # SIMPLIFIED VERSION ACTIONS
        # Legend:
            # Drone is centered (position 4)
            #     [1]    
            # [3] [0] [2]
            #     [4]    
        
        if action == 0:
            # WAIT
            new_pos = (drone.pos[0], drone.pos[1])
        elif action == 1:
            # Move to N
            new_pos = (drone.pos[0] - 1, drone.pos[1])
        elif action == 2:
            # Move to E
            new_pos = (drone.pos[0], drone.pos[1] + 1)
        elif action == 3:
            # Move to W
            new_pos = (drone.pos[0], drone.pos[1] - 1)
        elif action == 4:
            # Move to S
            new_pos = (drone.pos[0] + 1, drone.pos[1])
        else:
            # UNKNOWN ACTION - ERROR
            pass
        
        
        if (new_pos[0]<0 or new_pos[0]>=self.map_size[0] or new_pos[1]<0 or new_pos[1]>=self.map_size[1]):
            # OUT OF MAP 
            action_aborted = True
            pos_reward += self.reward_values.OUT_CELL
        elif (self.map[new_pos] == self.map_legend.CELL_FORBIDDEN):
            # FORBIDDEN CELL
            #action_aborted = True
            pos_reward += self.reward_values.FORBIDDEN_CELL
        else:
            # Check collisions
            collision = False
            for k in self.agents:
                if(self.agents[k].pos==new_pos and k != drone.id and (self.actives[k])):
                    collision = True
                    break
            if(collision):
                # COLLISION
                #action_aborted = True
                collision_reward += self.reward_values.COLLISION
        
        if(action_aborted):
            #action = 4 # WAIT 
            action = 0 # WAIT IN SIMPLIFIED VERSION
        else:
            drone.pos = new_pos
            self.agents[drone.id] = drone
            obs_reward = self._update_observed_map(drone)
            self._update_drone_map(drone)
        
        energy_reward = self.reward_values.ENERGY_MATRIX[self.actions[drone.id], action]
        
        # Compose the total individual reward
        ind_reward = pos_reward + obs_reward + collision_reward + energy_reward        
        
        self.actions[drone.id] = action
        return ind_reward    
    
    def _update_observed_map(self, drone):
        obs_reward = 0
        
        observed_pos = []
        obs_size = 2 * drone.view_range - 1
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + drone.pos[0] - drone.view_range + 1
                y = j + drone.pos[1] - drone.view_range + 1

                if(x<0 or x>=self.map_size[0] or y<0 or y>=self.map_size[1]):
                    # No valid observation, is out of map
                    continue
                
                # Circular view, we have to check if (x^2 + y^2 <= r^2)
                if ((drone.view_range - 1 - i)*(drone.view_range - 1 - i)+(drone.view_range - 1 - j)*(drone.view_range - 1 - j) > (drone.view_range*drone.view_range)):
                    # No valid observation, is out of the circle of vision
                    continue
                
                observed_pos.append((x,y))
        
        for pos in observed_pos:
            if (self.map[pos] == self.map_legend.CELL_TARGET_UNOBS):
                obs_reward += self.reward_values.UNOBSERVED_TARGET_CELL

                # REWARD SHAPING
#                 count_unobserved_cells = self.map[self.map==self.map_legend.CELL_TARGET_UNOBS].shape[0]
#                 count_observed_cells = self.map[self.map==self.map_legend.CELL_TARGET_OBS].shape[0]
#                 percentage_delta = count_observed_cells / (count_unobserved_cells + count_observed_cells)
#                 partial_completage_reward = int(percentage_delta * self.reward_values.UNOBSERVED_TARGET_CELL_DELTA)
#                 obs_reward += partial_completage_reward
                                
                self.map[pos] = self.map_legend.CELL_TARGET_OBS
                self.observed_map[pos] = self.map_legend.TARGET_OBS
            
        return obs_reward
    
    def _update_drone_map(self, drone):
        self.drone_map[np.where(self.drone_map == drone.map_id)] = 0
        self.drone_map[drone.pos] = drone.map_id
        
        self.current_dron_map = np.ones(self.map_size, dtype=np.int) * self.map_legend.NO_DRON
        self.current_dron_map[np.where(self.drone_map == self.agents[self.id].map_id)] = self.map_legend.CURRENT_DRON
        
        self.other_dron_map = np.ones(self.map_size, dtype=np.int) * self.map_legend.NO_DRON
        self.other_dron_map[np.where(self.drone_map != 0)] = self.map_legend.OTHER_DRON
        self.other_dron_map[np.where(self.drone_map == self.agents[self.id].map_id)] = self.map_legend.NO_DRON

        return
    
    def step(self, action): 
        self.steps += 1
        
        total_reward = 0     
        
        ind_reward = self._drone_step(self.agents[self.id], action)
        total_reward = ind_reward
        
        self.state[0] = self.obstacles_map
        self.state[1] = self.observed_map
        self.state[2] = self.current_dron_map
        if(self.drone_num > 1):
            self.state[3] = self.other_dron_map     # MULTIAGENT
        
        self.shared_state[0] = self.obstacles_map
        self.shared_state[1] = self.observed_map
        self.shared_state[2] = self.drone_map
        
        done = False
        
        if self._task_failed():
            done = True
            print(f'Thread {self.id} lost after {self.steps} steps')
        elif self._task_completed():
            total_reward += self.reward_values.DONE 
            done = True
            print(f'Thread {self.id} has reached GOAL for completing the map after {self.steps} steps, HURRAY!!!')
        elif(self.steps >= self.max_steps):
            total_reward += self.reward_values.NOT_DONE 
            done = True
            print(f'Thread {self.id} finished after reach maximum of steps')
            
        if done:
            total_reward += self.reward_values.REMAINING_TARGET_CELL * self.getUnobservedTargetCells()

        info = {}

        
        self.rewards[self.id] = float(total_reward)
        self.dones[self.id] = done
        
        self._save_environment()
   

        if (self.turn):
            self._release()
        if not (self.turn):
            self._acquire()
        
        if self.turn:
            self._load_environment()
            self._update_state()
            
            # Unmark reset_completed in the first iterations
            if(self.main_thread):
                if (False in self.resets.values()):
                    for k in self.resets:
                        if (not self.actives[k]):
                            self.resets[k] = True
                if (not False in self.resets.values()):
                    self.resets = {self.agents[k].id: False for k in self.agents}
                    self._unmark_reset() 
            
            # Prepare for finishing and reset the environment
            if True in self.dones.values():
                if not (self.dones[self.id]): 
                    if self._task_completed(): 
                        self.rewards[self.id] += self.reward_values.DONE
                    self.rewards[self.id] += self.reward_values.REMAINING_TARGET_CELL * self.getUnobservedTargetCells()
                    
                    self.dones[self.id] = True
                    self._save_environment()
                if self.main_thread:
                    while True:
                        wait = False
                        for i in range(1, self.drone_num + 1):
                            agent = (self.id + i) % self.drone_num
                            if (self.actives[agent] and not self.dones[agent]):
                                wait = True
                                break
                        if wait:
                            if (self.turn):
                                self._release()
                            if not (self.turn):
                                self._acquire()
                            self._load_environment()
                            self._update_state()
                        else:
                            break
                if (self.turn):
                    self._release()   
        else:
            # ERROR?
            print(f'Thread {self.id} fails to get the turn...')
            pass
        
        return (self.state, self.rewards[self.id], self.dones[self.id], info)  # SIMPLIFIED VERSION       
    

    
    ############################## EVALUATION METHODS ##############################
    def getSteps(self):
        return self.steps
    
    def getTargetCells(self):
        targetCells = 0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if (self.map[i, j] == self.map_legend.CELL_TARGET_OBS) or (self.map[i, j] == self.map_legend.CELL_TARGET_UNOBS):
                    targetCells += 1
        return targetCells
    
    def getObservedTargetCells(self):
        observedTargetCells = 0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if (self.map[i, j] == self.map_legend.CELL_TARGET_OBS):
                    observedTargetCells += 1
        return observedTargetCells
    
    def getUnobservedTargetCells(self):
        unobservedTargetCells = 0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if (self.map[i, j] == self.map_legend.CELL_TARGET_UNOBS):
                    unobservedTargetCells += 1
        return unobservedTargetCells
    
    def getObsevedCellRate(self):
        return (self.getObservedTargetCells() / self.getTargetCells())
    
    
    ################################ RENDER METHODS ################################
    def render(self, mode='human'):
        # By default, do nothing.
        pass

        
    # -------------------------------------------- VISUALIZATION METHODS --------------------------------------------
    
    def get_drone_obs(self, drone):
        obs_size = 2 * drone.view_range - 1
        obs = np.ones((obs_size, obs_size, 3))
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + drone.pos[0] - drone.view_range + 1
                y = j + drone.pos[1] - drone.view_range + 1

                
                if (drone.view_range - 1 - i)*(drone.view_range - 1 - i)+(drone.view_range - 1 - j)*(drone.view_range - 1 - j) > drone.view_range*drone.view_range:
                    obs[i, j, 0] = 0.5
                    obs[i, j, 1] = 0.5
                    obs[i, j, 2] = 0.5
                elif x>=0 and x<=self.map_size[0]-1 and y>=0 and y<=self.map_size[1]-1:
                    if self.map[x, y] == self.map_legend.CELL_EMPTY:
                        obs[i, j, 0] = 0
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0    
                    if self.map[x, y] == self.map_legend.CELL_TARGET_OBS:
                        obs[i, j, 0] = 0
                        obs[i, j, 1] = 1
                        obs[i, j, 2] = 0
                    if self.map[x, y] == self.map_legend.CELL_FORBIDDEN:
                        obs[i, j, 0] = 1
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0
                else:
                    obs[i, j, 0] = 0.5
                    obs[i, j, 1] = 0.5
                    obs[i, j, 2] = 0.5
        return obs

    def get_joint_obs(self):
        obs = np.ones((self.map_size[0], self.map_size[1], 3))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                obs[i, j, 0] = 0.5
                obs[i, j, 1] = 0.5
                obs[i, j, 2] = 0.5
        for k in self.agents:
            if self.actives[k]:
                temp = self.get_drone_obs(self.agents[k])

                for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        x = i + self.agents[k].pos[0] - self.agents[k].view_range + 1
                        y = j + self.agents[k].pos[1] - self.agents[k].view_range + 1
                        if_obs = True
                        if temp[i, j, 0] == 0.5 and temp[i, j, 1] == 0.5 and temp[i, j, 2] == 0.5:
                            if_obs = False
                        if if_obs == True:
                            obs[x, y, 0] = temp[i, j, 0]
                            obs[x, y, 1] = temp[i, j, 1]
                            obs[x, y, 2] = temp[i, j, 2]
        return obs
    
    
    def get_full_obs(self):
        obs = np.ones((self.map_size[0], self.map_size[1], 3))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map[i, j] == self.map_legend.CELL_TARGET_OBS:
                    obs[i, j, 0] = 0
                    obs[i, j, 1] = 1
                    obs[i, j, 2] = 0
                elif self.map[i, j] == self.map_legend.CELL_TARGET_UNOBS:
                    obs[i, j, 0] = 1
                    obs[i, j, 1] = 1
                    obs[i, j, 2] = 0
                elif self.map[i, j] == self.map_legend.CELL_FORBIDDEN:
                    obs[i, j, 0] = 1
                    obs[i, j, 1] = 0
                    obs[i, j, 2] = 0 

        for k in self.agents:
            if(self.actives[k]):
                obs[self.agents[k].pos[0], self.agents[k].pos[1], 0] = 0
                obs[self.agents[k].pos[0], self.agents[k].pos[1], 1] = 0
                obs[self.agents[k].pos[0], self.agents[k].pos[1], 2] = 1
        
        return obs