import os

import gym
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from omegaconf import open_dict
import habitat
import habitat.gym
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)  # Play a teaser video

from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    MeasurementConfig,
    HabitatSimRGBSensorConfig,
    HabitatSimDepthSensorConfig,
)

from habitat_sim.utils import viz_utils as vut
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import random
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from gym import spaces
import importlib

import PIL
import pickle

import multiprocessing as mp


from habitat.config import read_write
from habitat.core.dataset import BaseEpisode, Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks.registration import make_task
from habitat.utils import profiling_wrapper

if TYPE_CHECKING:
    from omegaconf import DictConfig

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def insert_render_options(config, rank=None, split="train"):
    # Added settings to make rendering higher resolution for better visualization
    with habitat.config.read_write(config):
        config.habitat.simulator.concur_render = False
        # config.habitat.simulator.forward_step_size = 1.0
        # config.habitat.simulator.turn_angle = 10
        config.habitat.dataset.split = split
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        myrgb_config = HabitatSimRGBSensorConfig(height=224, width=224)
        mydepth_config = HabitatSimDepthSensorConfig(height=224, width=224)
        agent_config.sim_sensors.update(
            {"myrgb_sensor": myrgb_config, "mydepth_sensor": mydepth_config}
        )
        OmegaConf.set_struct(agent_config, True)
        OmegaConf.update(
            agent_config, "sim_sensors.myrgb_sensor.uuid", "myrgb", force_add=True
        )
        OmegaConf.update(
            agent_config, "sim_sensors.mydepth_sensor.uuid", "mydepth", force_add=True
        )
        if rank is not None:
            config.habitat.seed = config.habitat.seed + rank

    return config


# import torch.multiprocessing as mp
# import torch.distributed as dist


def custom_print(msg, *args, rank=None, **kwargs):
    if rank is not None:
        print("[Process %d] " % (rank), end="")
    print(msg, *args, **kwargs)


def main(rank, world_size, total_episodes=None, split="train"):
    used_episodes = []
    conf = insert_render_options(
        habitat.get_config(
            "benchmark/nav/pointnav/pointnav_hm3d.yaml",
        ),
        split=split,
    )
    custom_print("Using seed %d" % (conf.habitat.seed), rank=rank)
    random.seed(conf.habitat.seed)
    np.random.seed(conf.habitat.seed)

    if total_episodes is None:
        total_episodes = 512000
    sub_episodes = total_episodes // world_size
    index_start = rank * sub_episodes
    index_stop = (rank + 1) * sub_episodes
    if rank == world_size - 1:
        index_stop = total_episodes
    custom_print("Start %d, stop %d" % (index_start, index_stop), rank=rank)

    env = habitat.Env(config=conf, index_start=index_start, index_stop=index_stop)
    # note that (scene_id, episode_id) is unique
    with open("episodes%d.pickle" % (rank), "wb") as f:
        pickle.dump([hash(v.scene_id + v.episode_id) for v in env.episodes], f)

    # custom_print("Finish setting env", rank=rank)

    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = conf.habitat.simulator.forward_step_size
    follower = ShortestPathFollower(env.sim, goal_radius, False)
    # print("Process %d ranges from %d to %d."%(rank, index_start, index_stop))
    # convert to meter, then convert to millimeter
    depth_scale = 10 * 1000.0
    ROOT = "data/hm3d_rgbd/%s" % (split)
    bar = tqdm(
        range(index_start, index_stop),
        leave=False,
        position=rank,
        dynamic_ncols=True,
        desc="Rank %d" % (rank),
    )
    all_steps = []
    for index in bar:
        count_steps = 0
        # Must reset for switching to a new episode
        observations = env.reset()  # noqa: F841
        ep_id = hash(env.current_episode.episode_id + env.current_episode.episode_id)
        used_episodes.append(ep_id)
        images = []
        depths = []
        while not env.episode_over:
            images.append(observations["myrgb"])
            depths.append(observations["mydepth"])
            # if count_steps==total_steps-1:
            #     a = {'action': 'stop', 'action_args': None}
            # else:
            #     a = np.random.choice(all_actions, p=action_probs)
            best_action = follower.get_next_action(
                env.current_episode.goals[0].position
            )
            if best_action is None:
                break
            observations = env.step(best_action)  # noqa: F841

            count_steps += 1
        all_steps.append(count_steps)
        bar.set_postfix({"idx": index, "avg": np.mean(all_steps)})
        SUB_ROOT = str(index // 10000)
        assert len(images) == len(depths)
        for i in range(len(images)):
            rgbi = images[i]
            rgbi = Image.fromarray(rgbi)
            path = os.path.join(ROOT, SUB_ROOT, str(index), "rgb", str(i * 10) + ".jpg")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            rgbi.save(path)
        for i in range(len(depths)):
            depthi = depths[i]
            depthi = (depthi * depth_scale).astype(np.int32).squeeze()
            depthi = Image.fromarray(depthi)
            path = os.path.join(
                ROOT, SUB_ROOT, str(index), "depth", str(i * 10) + ".png"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            depthi.save(path)
    with open("data/hm3d_rgbd/used_episodes%d.pickle" % (rank), "wb") as f:
        pickle.dump(used_episodes, f)


def main_with_range(index_start, index_stop):
    """Complete missing episodes, single thread"""
    used_episodes = []
    conf = insert_render_options(
        habitat.get_config(
            "benchmark/nav/pointnav/pointnav_hm3d.yaml",
        )
    )
    print("Using seed %d" % (conf.habitat.seed))
    random.seed(conf.habitat.seed)
    np.random.seed(conf.habitat.seed)

    print("Start %d, stop %d" % (index_start, index_stop))

    env = habitat.Env(config=conf, index_start=index_start, index_stop=index_stop)
    # note that (scene_id, episode_id) is unique

    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = conf.habitat.simulator.forward_step_size
    follower = ShortestPathFollower(env.sim, goal_radius, False)
    # print("Process %d ranges from %d to %d."%(rank, index_start, index_stop))
    # convert to meter, then convert to millimeter
    depth_scale = 1000.0 * 10
    ROOT = "data/hm3d_rgbd/train"
    bar = tqdm(
        range(index_start, index_stop),
        leave=False,
        position=0,
        dynamic_ncols=True,
        desc="Rank %d" % (0),
    )
    all_steps = []
    for index in bar:
        count_steps = 0
        # Must reset for switching to a new episode
        observations = env.reset()  # noqa: F841
        ep_id = hash(env.current_episode.episode_id + env.current_episode.episode_id)
        used_episodes.append(ep_id)
        images = []
        depths = []
        while not env.episode_over:
            images.append(observations["myrgb"])
            depths.append(observations["mydepth"])
            # if count_steps==total_steps-1:
            #     a = {'action': 'stop', 'action_args': None}
            # else:
            #     a = np.random.choice(all_actions, p=action_probs)
            best_action = follower.get_next_action(
                env.current_episode.goals[0].position
            )
            if best_action is None:
                break
            observations = env.step(best_action)  # noqa: F841

            count_steps += 1
        all_steps.append(count_steps)
        bar.set_postfix({"idx": index, "avg": np.mean(all_steps)})
        SUB_ROOT = str(index // 10000)
        assert len(images) == len(depths)
        for i in range(len(images)):
            rgbi = images[i]
            rgbi = Image.fromarray(rgbi)
            path = os.path.join(ROOT, SUB_ROOT, str(index), "rgb", str(i * 10) + ".jpg")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            rgbi.save(path)
        for i in range(len(depths)):
            depthi = depths[i]
            depthi = (depthi * depth_scale).astype(np.int32).squeeze()
            depthi = Image.fromarray(depthi)
            path = os.path.join(
                ROOT, SUB_ROOT, str(index), "depth", str(i * 10) + ".png"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            depthi.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--world_size",
        type=int,
        required=True,
        default=4,
    )
    parser.add_argument(
        "--total_episodes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    args = parser.parse_args()
    # mention the memory cost, which limits the number of threads
    if args.rank >= 0:  # external multi-processing
        assert args.rank < args.world_size, "Invalid rank!"
        main(
            rank=args.rank,
            world_size=args.world_size,
            total_episodes=args.total_episodes,
            split=args.split,
        )
    else:  # internal multi-processing
        simple_pool = []
        for i in range(args.world_size):
            p = mp.Process(
                target=main, args=(i, args.world_size, args.total_episodes, args.split)
            )
            p.start()
            simple_pool.append(p)
        for p in simple_pool:
            p.join()
    print("All finished")
