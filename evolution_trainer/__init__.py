"""evolution_trainer: the Unit B evolutionary process over dynamic_env.

A genome is the weight vector of a small TARGET-CONDITIONED policy network whose
input is ``env.state_vector(state)`` (objectives + target + structure) plus a
subgoal mask, and whose output is one action logit per legal action. A population
of agents pays a step cost per action, earns intermediate rewards for objective
progress, reproduces when two of the best share a reproduction fee (their child
mixes both parents' weights), spawns depth-limited subagents on intermediate
subgoals, and checkpoints the best agent of every generation.

Quick use::

    from evolution_trainer import EvoConfig, EvolutionTrainer
    from dynamic_env import load_spec

    env = load_spec("data/dynamic_env/spec_dynamic_group.json")
    trainer = EvolutionTrainer(env, EvoConfig(generations=20), spec_id="spec_dynamic_group")
    history = trainer.run()
"""

from .evolution import Agent, EpisodeResult, EvoConfig, EvolutionTrainer
from .features import ActionFeaturizer, state_dim, state_features
from .genome import MixResult, PolicyNet, mix_genomes, mutate_genome
from .harness import Harness, RolloutResult
from .mix_train import MixTrainer
from .rl_train import ReinforceTrainer

__all__ = [
    "ActionFeaturizer",
    "Agent",
    "EpisodeResult",
    "EvoConfig",
    "EvolutionTrainer",
    "Harness",
    "MixResult",
    "MixTrainer",
    "PolicyNet",
    "ReinforceTrainer",
    "RolloutResult",
    "mix_genomes",
    "mutate_genome",
    "state_dim",
    "state_features",
]
