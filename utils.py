import numpy as np
import pandas as pd
import gym

def word2grid(coord:tuple,
              data:pd.DataFrame,
              grid_shape:tuple):
    
    x, y = coord
    data_x = np.unique([data['PU_x'],data['DO_x']])
    data_y = np.unique([data['PU_y'],data['DO_y']])
    x_grid = int(grid_shape[0]*(x-data_x.min())/(data_x.max()-data_x.min()))
    y_grid = int(grid_shape[1]*(y-data_y.min())/(data_y.max()-data_y.min()))
    return x_grid,y_grid

# data['PU_grid_coor'] = data.apply(lambda x: word2grid((x.PU_x,x.PU_y),data,grid_shape),axis=1)
# data['DO_grid_coor'] = data.apply(lambda x: word2grid((x.DO_x,x.DO_y),data,grid_shape),axis=1)

def create_q_table(env):
    nA = env.action_space.n
    if isinstance(env.observation_space,gym.spaces.Box):
        num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    elif isinstance(env.observation_space,gym.spaces.Discrete):
        num_box = (env.observation_space.n,)
    return np.zeros(num_box + (nA,)), isinstance(env.observation_space,gym.spaces.Box)