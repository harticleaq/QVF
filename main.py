# -*- coding: utf-8 -*-
"""
@Author: Anqi Huang
@Time: 2024/3/8
"""

import os
import sys
import yaml
import numpy as np
import torch
import sys
sys.path.append('/home/aqh/haq_pro/MA/QQVF')
from os.path import dirname, abspath
from copy import deepcopy
from utils.logging import get_logger

from runner import run

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
os.environ["USE_TF"] = 'None'

# SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment("QVF")
# ex = Experiment("test")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(abspath(__file__)), "results")
if os.path.exists(results_path) is None:
    os.makedirs(results_path)

def addConfig(cons, con):
    for k, v in con.items():
        cons[k] = v
    return cons

def getConfig(arg_name, subfolder, params=None):
    with open(os.path.join(dirname(__file__), "config", subfolder, f"{arg_name}.yaml"), "r") as f:
        try:
            config = yaml.load(f,Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            f"{arg_name} yaml error: {e}!"
    return config


@ex.main
def my_main(_run, _config, _log):
    config = deepcopy(_config)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    run(_run, config, _log)


def start_proc(config, map):
    for i in range(2, 3):
        config["seed"] = i
        config["env_args"]["map_name"] = map
        ex.add_config(config)
        logger.info("Saving to sacred!")
        file_obs_path = os.path.join(results_path, "sacred")
        ex.observers.append(FileStorageObserver.create(file_obs_path))
        ex.run_commandline()

if __name__ == "__main__":
    params = sys.argv
    with open(os.path.join(dirname(__file__), "config", "common.yaml"), "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            f"config error: {e}"

    env_config = getConfig("sc2", "envs")
    alg_config = getConfig("qvf", "algs")
    config['env_args']['seed'] = config["seed"]


    config = addConfig(config, env_config)
    config = addConfig(config, alg_config)

    model_path = os.path.join(dirname(__file__), "models", config["alg"], config["env_args"]["map_name"])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    config["model_path"] = model_path
    ex.observers.append(MongoObserver.create("mongodb://192.168.1.102:27017", "sacred"))
    maps = ["3s5z_vs_3s6z", "bane_vs_bane"]
    # maps = ["3m"]
    for i in range(len(maps)):
        start_proc(config, maps[i])


    start_proc(config, maps[0])


