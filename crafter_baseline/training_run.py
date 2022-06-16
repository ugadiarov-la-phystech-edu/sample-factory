import collections
import json
from argparse import Namespace
from pathlib import Path

import gym
import numpy as np
import yaml
from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES, EXTRA_EPISODIC_STATS_PROCESSING
import crafter
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper
from sample_factory.envs.env_wrappers import PixelFormatChwWrapper

from sample_factory.utils.utils import log, static_vars

import wandb
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm

import sys
from utils.config_validation import Experiment
from wrappers import CrafterStats, compute_scores, UNLOCK_PREFIX


def make_crafter(full_env_name, cfg=None, env_config=None):
    env = gym.make('CrafterReward-v1')
    env = PixelFormatChwWrapper(env)
    env = CrafterStats(env, )
    env = MultiAgentWrapper(env)

    return env


@static_vars(cumulative_statistics=collections.defaultdict(collections.Counter))
def crafter_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    cumulative_statistics = crafter_extra_episodic_stats_processing.cumulative_statistics
    if stat_key == 'done':
        cumulative_statistics[policy_id]['n_episodes'] += 1
    elif stat_key.startswith(UNLOCK_PREFIX):
        cumulative_statistics[policy_id][stat_key] += stat_value


def crafter_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    cumulative_statistics = crafter_extra_episodic_stats_processing.cumulative_statistics
    if policy_id not in cumulative_statistics:
        return

    n_episodes = cumulative_statistics[policy_id]['n_episodes']
    if n_episodes == 0:
        return

    success_rates = []
    for key, value in cumulative_statistics[policy_id].items():
        if key.startswith(UNLOCK_PREFIX):
            success_rate = value / n_episodes
            summary_writer.add_scalar(f'success_rate_{key[len(UNLOCK_PREFIX):]}', success_rate, env_steps)
            success_rates.append(success_rate)

    score = compute_scores(np.asarray(success_rates) * 100)
    summary_writer.add_scalar('score', score, env_steps)
    log.debug(f'score: {round(float(score), 3)}')

    reward_mean = np.mean(policy_avg_stats['reward'])
    log.debug(f'reward_mean: {round(float(reward_mean), 3)}')
    summary_writer.add_scalar('reward_mean', reward_mean, env_steps)

    length_mean = np.mean(policy_avg_stats['len'])
    log.debug(f'length_mean: {round(float(length_mean), 3)}')
    summary_writer.add_scalar('length_mean', length_mean, env_steps)


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='CrafterReward-v1',
        make_env_func=make_crafter,
    )

    EXTRA_PER_POLICY_SUMMARIES.append(crafter_extra_summaries)
    EXTRA_EPISODIC_STATS_PROCESSING.append(crafter_extra_episodic_stats_processing)


def validate_config(config):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            full_config=exp.dict()
                            )
    return exp, flat_config


def main():
    register_custom_components()

    import argparse

    parser = argparse.ArgumentParser(description='Process training config.')

    parser.add_argument('--config_path', type=str, action="store",
                        help='path to yaml file with single run configuration', required=False)

    parser.add_argument('--raw_config', type=str, action='store',
                        help='raw json config', required=False)

    parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                        help='Run wandb in thread mode. Usefull for some setups.', required=False)

    params = parser.parse_args()

    if params.raw_config:
        config = json.loads(params.raw_config)
    else:
        if params.config_path is None:
            config = Experiment().dict()
        else:
            with open(params.config_path, "r") as f:
                config = yaml.safe_load(f)

    exp, flat_config = validate_config(config)
    log.debug(exp.global_settings.experiments_root)

    if exp.global_settings.use_wandb:
        import os
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project=exp.name, config=exp.dict(), save_code=False, sync_tensorboard=True)

    status = run_algorithm(flat_config)
    if exp.global_settings.use_wandb:
        import shutil
        path = Path(exp.global_settings.train_dir) / exp.global_settings.experiments_root
        zip_name = str(path)
        shutil.make_archive(zip_name, 'zip', path)
        wandb.save(zip_name + '.zip')
    return status


if __name__ == '__main__':
    sys.exit(main())
