#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher, postprocess_observation
from models import Encoder, ObservationModel, RewardModel, TransitionModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cheetah-run', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
    parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
    parser.add_argument('--action-repeat', type=int, default=4, metavar='R', help='Action repeat')
    parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
    parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
    parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')
    parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
    parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
    parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F), help='Model activation function')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of timesteps to collect')
    parser.add_argument('--models', type=str, required=True, help='Absolute path to model checkpoint')
    device = torch.device('cpu')
    args = parser.parse_args()

    # Setup the dm_control env
    env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length,
              args.action_repeat, args.bit_depth)
    init_observation = env.reset()
    
    # Load the models
    model_dicts = torch.load(args.models, map_location='cpu')
    transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size, args.embedding_size, args.activation_function).to(device=device)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size, args.state_size, args.embedding_size, args.activation_function).to(device=device)
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.activation_function).to(device=device)
    reward_model.load_state_dict(model_dicts['reward_model'])
    encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.activation_function).to(device=device)
    encoder.load_state_dict(model_dicts['encoder'])
  

    observations = torch.zeros(args.n_steps, 3, 64, 64)
    actions = torch.zeros(args.n_steps, env.action_size)
    for i in range(args.n_steps):
        action = env.sample_random_action()
        observation, reward, done = env.step(action)
        observations[i] = observation
        actions[i] = action

    with torch.no_grad():
        belief = torch.zeros(1, args.belief_size, device=device)
        posterior_state = torch.zeros(1, args.state_size, device=device)
        # Adding time dimension to these:
        action = torch.zeros(1, env.action_size, device=device)
        # observation = observations[0].unsqueeze(dim=0)
        
        # Get initial belief and state posterior using the first observation
        belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state,
                                                                  action.unsqueeze(dim=0),
                                                                  belief,
                                                                  encoder(init_observation).unsqueeze(dim=0))
        belief = belief.squeeze(dim=0) # Remove time
        posterior_state = posterior_state.squeeze(dim=0) # Remove time
        
        # Setting prior as initial posterior
        prior_state = posterior_state
        for i in range(args.n_steps):
            action = actions[i].unsqueeze(dim=0)
            
            belief, prior_state, _, _ = transition_model(prior_state,
                                                         action.unsqueeze(dim=0),
                                                         belief)

            belief = belief.squeeze(dim=1)
            prior_state = prior_state.squeeze(dim=1)
            
            decoded = observation_model(belief, prior_state).cpu().numpy().squeeze()
            decoded = postprocess_observation(decoded, args.bit_depth)
            decoded = np.transpose(decoded, (1, 2, 0))
            decoded = cv2.resize(decoded, (512, 512))

            actual = observations[i].numpy().transpose(1, 2, 0)
            actual = postprocess_observation(actual, args.bit_depth)
            actual = cv2.resize(actual, (512, 512))
            
            img = np.concatenate([actual, decoded], axis=1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('screen', img)
            cv2.waitKey(0)
            
