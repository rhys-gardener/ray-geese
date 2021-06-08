import argparse
from gym.spaces import Discrete
import os
import random

import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer, PGTFPolicy
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.examples.policy.rock_paper_scissors_dummies import \
    BeatLastHeuristic, AlwaysSameHeuristic
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from EnvWrapperRay import HungryGeeseKaggle

from ray.rllib.models.utils import get_filter_config
tf1, tf, tfv = try_import_tf()
import logging
logging.basicConfig(filename='logging2.log', level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=150,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=1000.0,
    help="Reward at which we stop training.")


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


def run_same_policy(args, stop):
    """Use the same policy for all agents (trivial case)."""
    config = {
        "env": HungryGeeseKaggle,
        "framework": args.framework,
        'train_batch_size': 512,
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": False,
        },
        "no_done_at_end": True
    }

    results = tune.run("PPO", config=config, stop=stop, verbose=1)
    print(results.best_result_df)

    if args.as_test:
        # Check vs 0.0 as we are playing a zero-sum game.
        check_learning_achieved(results, 10000.0)




def main():
    args = parser.parse_args()
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    run_same_policy(args, stop=stop)
    print("run_same_policy: ok.")


if __name__ == "__main__":
    main()