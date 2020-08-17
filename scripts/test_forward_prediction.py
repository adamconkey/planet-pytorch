#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cheetah-run', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
    parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
    parser.add_argument('--action-repeat', type=int, default=4, metavar='R', help='Action repeat')
    parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
    parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
    parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of timesteps to collect')
    parser.add_argument('--models', type=str, required=True, help='Absolute path to model checkpoint')
    device = torch.device('cpu')
    args = parser.parse_args()

    # Load the models
    models = torch.load(args.models)
    print(models.keys())
    sys.exit(0)

    # Setup the dm_control env
    env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
    env.reset()

    observations = np.zeros((args.n_steps, 3, 64, 64))
    actions = np.zeros((args.n_steps, env.action_size))
    for i in range(args.n_steps):
        action = env.sample_random_action()
        observation, reward, done = env.step(action)
        observations[i] = observation.numpy()
        actions[i] = action.numpy()

    with torch.no_grad():
        belief = torch.zeros(1, args.belief_size, device=device)
        posterior_state = torch.zeros(1, args.state_size, device=device)
        action = torch.zeros(1, env.action_size, device=device)
        observation = observations[0] # Using first observation to the try to predict forward through latent
        
        # Get initial belief and state posterior using the first observation
        belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state,
                                                                  action.unsqueeze(dim=0),
                                                                  belief,
                                                                  encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
        belief = belief.squeeze(dim=0) # Remove time
        posterior_state = posterior_state.squeeze(dim=0) # Remove time

        # Apply actions propagating through latent space
        for i in range(args.n_steps):
            action = actions[i]
            belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state,
                                                                      action.unsqueeze(dim=0),
                                                                      belief)
            
            decoded = observation_model(belief, posterior_state).cpu().numpy()
            
            print(decoded.shape)
