import gym
import numpy as np


class Multi_passanger_taxi(gym.Env):
    def __init__(self,env_config):
        ## Get data from input
        self.data = env_config['demand_data']
        self.distance_data = env_config['distance_data']
        self.max_timestep =  env_config['max_timestep']
        self.n = env_config['customers_per_taxi'] # max number of customer per taxi

        self.locations =  np.unique(self.distance_data.PULocationID.values)
        self.nb_locations = len(self.locations)

        self.locations = dict(zip(range(self.nb_locations),self.locations))
        self.location_to_index = dict((v, k) for k, v in self.locations.items())

        self.customers=0
        self.custumer_count = self.data.Demand.sum() 
        self.customers_limit = min(self.custumer_count,env_config['client_limit'])
        # self.taxi_space=taxi_space
        # Define Actions we can 
        # self.action_space = gym.spaces.Discrete(self.nb_locations*2)
        
        self.action_space = gym.spaces.Dict({'move':gym.spaces.Discrete(self.nb_locations),
                                            'pickup' : gym.spaces.Discrete(2)})
        # self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(self.nb_locations),
        #                                     gym.spaces.Discrete(2)])

        # self.action_space = gym.spaces.MultiDiscrete([self.nb_locations,2])
        

        # Define Observation space 
        spaces = {'position': gym.spaces.Box(low = np.array([0]),high=np.array([self.nb_locations-1]),dtype=int),
                  'state':gym.spaces.Box(low = -1,high=self.nb_locations-1,shape=(self.n,),dtype=int),}
        
        self.observation_space = gym.spaces.Dict(spaces)
        # self.action_space_transformatiom()
        
#         self.observation_space =gym.spaces.Box(low = np.array([0,-1]), 
#                high=np.array([self.nb_locations-1,self.nb_locations-1]),dtype=int)
        # ## Initialise the space  
        self.reset()    
            
#     def action_space_transformatiom(self):
#         action_space = self.action_space
#         shape = [space.n for space in action_space.__dict__['spaces'].values()]

#         list_actions = list(itertools.product(*[range(i) for i in shape]))
#         self.action_map = dict(zip(range(len(list_actions)),list_actions))
#         self.action_space = gym.spaces.Discrete(len(list_actions))
        
    def get_distane(self,PU_location,DO_location):
            distance = self.distance_data[(self.distance_data.PULocationID==PU_location) & 
                                          (self.distance_data.DOLocationID==DO_location)]['distance'].values
            return distance[0]

    def step(self, action):
        done = False
        reward =0
        move_action = action['move']
        pick_up_action= action['pickup']
        taxi_location = self.locations[self.state['position'][0]]
        taxi_state = self.state['state']

        if pick_up_action==1: ## If action is pick up client:
            # print(taxi_state.values())
            if -1 not in taxi_state: ## taxi is full
                reward -=100
                # print(self.state)
            else: ## taxi has an empty spot
                for i in range(self.n):
                    if taxi_state[i] == -1:
                        if taxi_location in self.demand_dict.keys(): ## Current location has a client:
                            print(f'Pick up client in position {i}')
                            destination_location = self.demand_dict[taxi_location].pop(0)
                            self.state['state'][i] = self.location_to_index[destination_location]
                            if self.demand_dict[taxi_location]==[]:
                                self.demand_dict.pop(taxi_location)
                            reward+=10
                            break 
                        else:  ## Taxi empty and location has no client:
                            reward+= 0
                            
        action_location = self.locations[move_action]
        # print(action_location, taxi_state)
        if action_location in taxi_state: ## if action is going to the client destination
            destination_indexes, = np.where(taxi_state==action_location)
            # print('npwhere res',destination_indexes)
            for destination_index in destination_indexes:
                # print('single index', destination_index)    
                # print('chennnn')
                self.customers+=1 
                self.state['state'][destination_index]=-1
                reward += self.get_distane(taxi_location,action_location)*100
        else: #We have a client but we are not going to his destination
            reward += -self.get_distane(taxi_location,action_location)*10

        
        self.taxi_path.append(move_action)
        self.state['position'][0] = move_action
        self.time_step +=1

        if self.customers == self.customers_limit or len(self.demand_dict)==0:
            done=True

        # Set placeholder for info
        info = {'taxi location & state':self.state,
               'nb of satistied customers ': self.customers
               }
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        self.time_step = 0
        self.taxi_path =[]
        self.state = self.observation_space.sample()
        self.state['state']=-np.ones(self.n)
        # print(self.state)
        self.demand_dict = {pu :[] for pu in self.data.PU_LocationID}
        for index in range(len(self.data)):
            row_index = self.data.loc[index]
            # print(row_index.PU_LocationID)
            self.demand_dict[row_index.PU_LocationID]+=[row_index.DO_LocationID for _ in range(row_index.Demand)]
        self.customers=0
        return self.state


class Single_passanger_taxi(gym.Env):
    def __init__(self,env_config):
        ## Get data from input
        self.data = env_config['demand_data']
        self.distance_data = env_config['distance_data']
        self.max_timestep =  env_config['max_timestep']

        self.locations =  np.unique(self.distance_data.PULocationID.values)
        self.nb_locations = len(self.locations)

        self.locations = dict(zip(range(self.nb_locations),self.locations))
        self.location_to_index = dict((v, k) for k, v in self.locations.items())

        self.customers=0
        self.custumer_count = self.data.Demand.sum() 
        self.customers_limit = min(self.custumer_count,env_config['client_limit'])
        # self.taxi_space=taxi_space
        # Define Actions we can 
        self.action_space = gym.spaces.Discrete(self.nb_locations*2)
        
        # self.action_space = gym.spaces.MultiDiscrete([self.nb_locations,2])

        # Define Observation space 
        self.observation_space =gym.spaces.Box(low = np.array([0,-1]), 
               high=np.array([self.nb_locations-1,self.nb_locations-1]),dtype=int) ##TODO: maybe repeated info here ????

        # ## Initialise the space  
        self.reset()

    def get_distane(self,PU_location,DO_location):
            distance = self.distance_data[(self.distance_data.PULocationID==PU_location) & 
                                          (self.distance_data.DOLocationID==DO_location)]['distance'].values
            return distance[0]
        
    def discretize_action_space(self,action):
        list_action = [(i,j) for i in range(self.nb_locations) for j in [0,1]]
        dict_action = dict(zip(range(2*self.nb_locations),list_action))
        return dict_action[action]
        
    def pick_up_action(self):
        taxi_location = self.locations[self.state[0]]
        if self.state[1] == -1: ##Check if taxi is empty
            if taxi_location in self.demand_dict.keys(): ## Current location has a client:
                destination_location = self.demand_dict[taxi_location].pop(0)
                self.state[1] = self.location_to_index[destination_location]
                if self.demand_dict[taxi_location]==[]:
                    self.demand_dict.pop(taxi_location)
                return 10
            else:  ## Taxi empty and location has no client:
                return 0
        else: ## Taxi has a client so it's impossible action to pick up one client
            return  -100

    def move_action(self,action):
            action_location = self.locations[action]
            taxi_location = self.locations[self.state[0]]
            if self.state[1] == -1: ## Empty taxi
                # return -self.get_distane(taxi_location,action_location)
                return 0 
            else: ## taxi already has a client 
                taxi_destination = self.locations[self.state[1]]
                if action == self.state[1]: ## if action is going to the client destination
                    self.customers+=1
                    self.state[1]= -1
                    return  self.get_distane(taxi_location,action_location)*100
                else: #We have a client but we are not going to his destination
                    return -self.get_distane(taxi_location,action_location)*10

    def step(self, action):
        
        action = self.discretize_action_space(action)
        # print(self.state)
        done = False
        reward =0
        go_action = action[0]
        pick_up_action= action[1]

        if pick_up_action==1: ## If action is pick up client:
            reward += self.pick_up_action()
        reward = self.move_action(go_action)
        
        self.taxi_path.append(action[0])
        self.state[0] = action[0] 
        self.time_step +=1

        if self.customers == self.customers_limit or len(self.demand_dict)==0:
            done=True

        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        self.time_step = 0
        self.taxi_path =[]
        self.state = self.observation_space.sample()
        # self.state[2]=0
        self.state[1]=-1
        self.demand_dict = {pu :[] for pu in self.data.PU_LocationID}
        for index in range(len(self.data)):
            row_index = self.data.loc[index]
            # print(row_index.PU_LocationID)
            self.demand_dict[row_index.PU_LocationID]+=[row_index.DO_LocationID for _ in range(row_index.Demand)]
        self.customers=0
        return self.state



class Env_first(gym.Env):
    ## Single customer, fully observable space
    ## One client per locations, we directly pick up the next client if taxi is empty
    def __init__(self,data,client_limit):
        self.data = data
        self.pick_up = list(set(self.data.PULocationID))
        self.drop_off = list(set(self.data.DOLocationID))
#         self.distance = self.data.distance
        self.locations =  list(set(self.pick_up+self.drop_off))
        self.nb_locations = len(self.locations)

        # Actions we can
        self.action_space = gym.spaces.Discrete(self.nb_locations)


        self.observation_space = gym.spaces.Box(low = np.zeros(self.nb_locations+1),
               high=np.array([self.nb_locations-1]+ [1 for _ in range(self.nb_locations)]),dtype=int)

        self.state = self.observation_space.sample()
        self.init_state()
        self.customers=0
        self.customers_limit = client_limit

    def init_state(self):
        for index_location in range(self.nb_locations):
            PU_location = self.locations[index_location]
            if PU_location in self.pick_up:
                self.state[index_location+1]=1
            else:
                self.state[index_location+1]=0

    def get_destinations(self,PU_location):
        if PU_location in self.pick_up:
            destinations = list(self.data[self.data.PULocationID==PU_location]['DOLocationID'])
            return destinations[0]
        else:
            print('Not a valid pick up location',PU_location, self.state[self.state[0]+1],self.state[0], self.locations[self.state[0]])
            return None

    def get_reward(self,PU_location,DO_location):
        PU_data = self.data[self.data.PULocationID==PU_location]
        if PU_location==DO_location:
            return 0
        if any(PU_data.DOLocationID==DO_location):
            distance = float(PU_data[PU_data.DOLocationID==DO_location]['distance'])
            return distance
        else:
            return 10

    def step(self, action):
        taxi_location_id = self.state[0]
        if taxi_location_id>= len(self.locations):
            print(taxi_location_id)
        taxi_location = self.locations[taxi_location_id]
        done = False

        if self.state[taxi_location_id+1]==0:
            reward = -1
        else:
            destination_target = self.get_destinations(taxi_location)
            destination_action = self.locations[action]
            distance =  self.get_reward(taxi_location,destination_action)
            if destination_action == destination_target:
                reward = 100
                self.state[taxi_location_id+1]=0
                self.customers+=1
#                 print('suctom count',self.customers)
            else:
                reward= 0

        self.state[0]=action

        if self.customers == self.customers_limit:
            done=True

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        self.state = self.observation_space.sample()
        self.init_state()
        self.customers=0
        return self.state
