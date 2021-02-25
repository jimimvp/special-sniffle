from operator import gt
import os
import numpy as np
from jax import numpy as jnp
import jax
from ..simulator import World, Simulator


def score_fn(finishing_times, config):
     
    score = np.zeros(len(finishing_times))
    for i, ft in enumerate(finishing_times):
        ft = np.array(ft)
        score[i] = config.car_score*len(ft)  + np.sum(config.simulation_duration - ft)
    
    return score


def simulate_sol(traffic_lights, config):
    world = World(config=config)
    simulator = Simulator(config=config, world=world)
    i = 0
    for _, intersection in world.intersections.items():
        intersection.traffic_lights = traffic_lights[i:i+len(intersection.in_streets)]
        i+=len(intersection.in_streets)
    
    simulator.start_loop(verbose=False)

    finishing_times = []
    for car in world.cars:
        if car.done:
            finishing_times.append(car.finish_time)



def sample_trajectories(traffic_lights, config):


    res = []
    for tf in traffic_lights:
        
        finishing_times = simulate_sol(traffic_lights, config)
        res.append(finishing_times)
    
    return res








def cem_improved(key, config, p, d, K, num_elites=10, mu = None, 
    std=None, sampling_f=lambda x:x, sampling_d=None, max_keep_elites=None, score_fn=score_fn, 
    act_low=None,act_high=None):
    
    if sampling_d is None:
        sampling_d = d

    elites = None

    # step 0: define initial distribution
    if act_low is None:
        act_low, act_high = -1, 1
    if mu is None:
        mu = jnp.ones((sampling_d))*(act_high+act_low)/2
    if std is None:
        std = jnp.ones((sampling_d))*1


    for _ in range(K):

        # step 1: sample initial trajectories from initial distribution
        keys = jax.random.split(key, 2)

        # laplacian sample
        actions = jax.random.laplace(keys[-1], (p,sampling_d))*std + mu
    
        actions = jnp.clip(actions, act_low, act_high)

        keys = jax.random.split(keys[-1])

        # transform actions in case of sampling from lower dimension
        actions_exec = sampling_f(actions)

        finishing_times = sample_trajectories(actions_exec, config)
        finishing_times = np.array(finishing_times)

        # check if previous elites contain perticularly good elites
        if max_keep_elites and not elites is None:

            l_prev = min(max_keep_elites, len(elites))
            prev_elites = elites[:l_prev]

            #print(prev_elites.shape, prev_elite_returns.shape, prev_elite_states.shape, prev_elite_next_obs.shape, prev_elite_costs.shape)
            actions = jnp.vstack([actions, prev_elites])
            finishing_times = np.hstack([finishing_times, prev_finishing_times])



        scores = score_fn(finishing_times, config)
        # step 2: calculate returns

        # step 3: get elites
        elite_idxs = (-scores).argsort()[:num_elites]

        print("Best elite:", scores[elite_idxs[0]])
        print("Worst elite:", scores[elite_idxs[-1]])
  

        elites = actions[elite_idxs]
        prev_finishing_times = finishing_times[elite_idxs]

        # step 4: refit initial distribution
        mu = jnp.mean(elites, axis=0)
        std = jnp.std(elites, axis=0)

        print(elites)

    return mu, std, sampling_f(elites[0])


