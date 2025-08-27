import jax.numpy as jnp
import jax.random as rnd
import pytest

from robotools.plan.trajectory import Trajectory
import robotools.plan.trajectory_optimization as trajopt
from robotools.model.library import DoubleIntegrator


@pytest.fixture
def dynamic_model():
    return DoubleIntegrator()


def test_direct_transcription(dynamic_model):
    num_steps = 10
    key = rnd.key(0)
    states = rnd.uniform(key, (num_steps + 1, dynamic_model.n))
    controls = rnd.uniform(key, (num_steps, dynamic_model.m))
    dt = 0.1

    trajectory = Trajectory(states, controls, dt)
    direct_transcription = trajopt.DirectTranscription(dynamic_model)
    direct_transcription.approximate(trajectory)
    assert True
