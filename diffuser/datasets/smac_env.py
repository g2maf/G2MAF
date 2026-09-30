import os
from typing import Any, Dict, List, Optional

import gym
import numpy as np

# SC2/SMAC map metadata (obs_size excludes agent_id)
SMAC_MAP_INFO = {
    '3m':       {'n_agents': 3, 'obs_size': 7,  'n_actions': 9,  'episode_limit': 60},
    '5m_vs_6m': {'n_agents': 5, 'obs_size': 72, 'n_actions': 14, 'episode_limit': 70},
    '2s3z':     {'n_agents': 5, 'obs_size': 80, 'n_actions': 16, 'episode_limit': 120},
    '8m':       {'n_agents': 8, 'obs_size': 7,  'n_actions': 14, 'episode_limit': 120},
}

_SC2_ENV_AVAILABLE = False
try:
    from smac.env import StarCraft2Env
    # Only mark available if SC2 binary exists
    import os as _os
    _sc2_path = _os.environ.get('SC2PATH', _os.path.expanduser('~/StarCraftII'))
    if _os.path.exists(_sc2_path):
        _SC2_ENV_AVAILABLE = True
except ImportError:
    pass


class SMACMock:
    """Minimal mock for StarCraft2Env when SC2 binary is not installed."""
    def __init__(self, map_name, obs_last_action=False):
        info = SMAC_MAP_INFO.get(map_name)
        if info is None:
            raise ValueError("Unknown SMAC map: {}. Known: {}".format(
                map_name, list(SMAC_MAP_INFO)))
        self.n_agents = info['n_agents']
        self.n_actions = info['n_actions']
        self.episode_limit = info['episode_limit']
        self._obs_size = info['obs_size']

    def get_obs_size(self):
        return self._obs_size

    def get_obs(self):
        return [np.zeros(self._obs_size) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def get_stats(self):
        return {}

    def reset(self):
        pass

    def step(self, actions):
        return 0.0, True, {}


class SMAC(gym.Env):
    """Environment wrapper SMAC."""

    metadata = {}

    def __init__(self, map_name, add_agent_ids_to_obs=True):
        if _SC2_ENV_AVAILABLE:
            self._environment = StarCraft2Env(map_name=map_name, obs_last_action=False)
        else:
            self._environment = SMACMock(map_name)
        self._agents = ["agent_{}".format(n) for n in range(self._environment.n_agents)]
        self.num_agents = len(self._agents)
        self.num_actions = self._environment.n_actions
        self._done = False
        self.max_episode_length = self._environment.episode_limit
        self.add_agent_ids_to_obs = add_agent_ids_to_obs

        if add_agent_ids_to_obs:
            self.one_hot_agent_ids = []
            for i in range(self.num_agents):
                agent_id = np.eye(self.num_agents)[i]
                self.one_hot_agent_ids.append(agent_id)
            self.one_hot_agent_ids = np.stack(self.one_hot_agent_ids, axis=0)

        self.observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self._environment.get_obs_size() + self.num_agents
                    if add_agent_ids_to_obs
                    else self._environment.get_obs_size(),
                ),
            )
            for _ in range(self.num_agents)
        ]
        self.action_space = [
            gym.spaces.Discrete(n=self.num_actions) for _ in range(self.num_agents)
        ]

    def reset(self):
        self._environment.reset()
        self._done = False
        observation = np.array(self._environment.get_obs())
        if self.add_agent_ids_to_obs:
            observation = np.concatenate([self.one_hot_agent_ids, observation], axis=1)
        return observation

    def step(self, actions):
        reward, self._done, self._info = self._environment.step(actions)
        reward_n = np.array([reward for _ in range(self.num_agents)])
        done_n = np.array([self._done for _ in range(self.num_agents)])
        next_observation = np.array(self._environment.get_obs())
        if self.add_agent_ids_to_obs:
            next_observation = np.concatenate(
                [self.one_hot_agent_ids, next_observation], axis=1
            )
        return next_observation, reward_n, done_n, self._info

    def env_done(self):
        return self._done

    def get_legal_actions(self):
        legal_actions = []
        for i, _ in enumerate(self._agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return np.array(legal_actions)

    def get_stats(self):
        return self._environment.get_stats()

    @property
    def agents(self):
        return self._agents

    @property
    def possible_agents(self):
        return self._agents

    @property
    def environment(self):
        return self._environment

    def __getattr__(self, name):
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


def load_environment(name, **kwargs):
    if type(name) is not str:
        return name

    idx = name.find('-')
    env_name, data_split = name[:idx], name[idx + 1:]

    env = SMAC(env_name, **kwargs)
    if hasattr(env, 'metadata'):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata['data_split'] = data_split
    env.metadata['name'] = env_name
    env.metadata['global_feats'] = ['states']
    return env


def sequence_dataset(env, preprocess_fn):
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        'data/smac',
        env.metadata['name'],
        env.metadata['data_split'],
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError('Dataset directory not found: {}'.format(dataset_path))

    observations = np.load(os.path.join(dataset_path, 'obs.npy'))
    legal_actions = np.load(os.path.join(dataset_path, 'legals.npy'))
    rewards = np.load(os.path.join(dataset_path, 'rewards.npy'))
    actions = np.load(os.path.join(dataset_path, 'actions.npy'))
    path_lengths = np.load(os.path.join(dataset_path, 'path_lengths.npy'))

    start = 0
    for path_length in path_lengths:
        end = start + path_length
        episode_data = {}
        episode_data['observations'] = observations[start:end]
        episode_data['legal_actions'] = legal_actions[start:end]
        episode_data['rewards'] = rewards[start:end]
        episode_data['actions'] = actions[start:end]
        episode_data['terminals'] = np.zeros(
            (path_length, observations.shape[1]), dtype=bool
        )
        episode_data['terminals'][-1] = True
        yield episode_data
        start = end
