""" Contains the Episodes for Navigation. """
import random
import torch
import time
import math
import sys
from itertools import permutations
from constants import GOAL_SUCCESS_REWARD, INTERMED_FIND_REWARD, STEP_PENALTY, BASIC_ACTIONS
from environment import Environment
from utils.net_util import gpuify


class Episode:
    """ Episode for Navigation. """
    def __init__(self, args, gpu_id, rank, strict_done=False):
        super(Episode, self).__init__()

        self._env = None

        self.fail_penalty = args.failed_action_penalty
        self.consec_rotate_penalty_coeff = args.rotate_penalty
        self.lookup_penalty = args.lookup_penalty
        self.distance_penalty_coeff = args.distance_penalty

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None

        self.seed = args.seed + rank
        random.seed(self.seed)

        with open('./datasets/objects/int_objects.txt') as f:
            int_objects = [s.strip() for s in f.readlines()]
        with open('./datasets/objects/rec_objects.txt') as f:
            rec_objects = [s.strip() for s in f.readlines()]
        
        self.objects = int_objects + rec_objects

        self.actions_list = [{'action':a} for a in BASIC_ACTIONS]
        self.actions_taken = []

    @property
    def environment(self):
        return self._env

    def state_for_agent(self):
        return self.environment.current_frame

    def step(self, action_as_int):
        action = self.actions_list[action_as_int]
        self.actions_taken.append(action)
        return self.action_step(action)

    def action_step(self, action):
        self.environment.step(action)
        return self.judge(action)

    def slow_replay(self, delay=0.2):
        # Reset the episode
        self._env.reset(self.cur_scene, change_seed = False)
        
        for action in self.actions_taken:
            self.action_step(action)
            time.sleep(delay)

    def _target_path_distance(self):
        tgt_positions = [o['position'] for o in self._env.last_event.metadata['objects']
                         if o['objectType'] in self.remaining_targets]
        if len(tgt_positions) == 0:
            return 0.

        # This is very slow, but since we only have two objects at the moment, we'll
        # put up with it
        best_distance = None
        for tgt_path in permutations(tgt_positions, len(tgt_positions)):
            path = [self._env.last_event.metadata['agent']['position']] + list(tgt_path)
            total_distance = 0.
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                total_distance += math.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2)
            if best_distance is None or total_distance < best_distance:
                best_distance = total_distance
        return best_distance

    
    def judge(self, action):
        """ Judge the last event. """
        # immediate reward
        reward = STEP_PENALTY 
        done = False
        action_was_successful = self.environment.last_action_success
        shaped_rewards = {}
        
        shaped_rewards['fail_penalty'] = 0.
        if not action_was_successful:
            shaped_rewards['fail_penalty'] = self.fail_penalty
            reward += self.fail_penalty

        shaped_rewards['distance_penalty'] = self.distance_penalty_coeff * self._target_path_distance()
        reward += shaped_rewards['distance_penalty']

        if action['action'] in ('RotateLeft', 'RotateRight'):
            self.consecutive_rotates += 1
            consec_penalty = -self.consec_rotate_penalty_coeff * math.exp(self.consecutive_rotates)
            shaped_rewards['consec_penalty'] = consec_penalty
            reward += consec_penalty
        else:
            shaped_rewards['consec_penalty'] = 0.
            self.consecutive_rotates = 0

        horizon = self._env.last_event.metadata['agent']['cameraHorizon']
        shaped_rewards['lookup_penalty'] = 0.
        if not (horizon == 30 or horizon == 0):
            reward += self.lookup_penalty
            shaped_rewards['lookup_penalty'] = self.lookup_penalty

        if action['action'].startswith('Seen'):
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            
            seen_obj_name = action['action'][4:]
            if seen_obj_name in self.remaining_targets and seen_obj_name in visible_objects:
                self.remaining_targets.remove(seen_obj_name)
                if len(self.remaining_targets) == 0:
                    done = True
                    reward += GOAL_SUCCESS_REWARD
                    self.success = True
                else:
                    reward += INTERMED_FIND_REWARD

        return reward, done, action_was_successful, shaped_rewards

    def new_episode(self, args, scene):
        
        if self._env is None:
            if args.arch == 'osx':
                local_executable_path = './datasets/builds/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64'
            else:
                local_executable_path = './datasets/builds/thor-local-Linux64'
            
            self._env = Environment(
                    grid_size=args.grid_size,
                    fov=args.fov,
                    local_executable_path=local_executable_path,
                    randomize_objects=args.randomize_objects,
                    seed=self.seed)
            self._env.start(scene, self.gpu_id)
        else:
            self._env.reset(scene)

        # For now, single target.
        self.fail_penalty = args.failed_action_penalty
        self.lookup_penalty = args.lookup_penalty
        self.targets = ['Tomato', 'Bowl']
        self.remaining_targets = list(self.targets)
        self.success = False
        self.cur_scene = scene
        self.actions_taken = []
        self.consecutive_rotates = 0
        
        return True
