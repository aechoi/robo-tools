import jax.numpy as jnp
import pytest

import robotools.model.dynamics as dyn
from robotools.model.library import DoubleIntegrator


@pytest.fixture
def test_dyn():
    return DoubleIntegrator()


def test_discrete_simulate(test_dyn):
    init_state = jnp.array([0.0, 0.0])

    def policy(state, t):
        return jnp.array([t])

    num_steps = 10
    dt = 0.1

    state_traj, control_traj = test_dyn.discrete_simulate(
        init_state, policy, num_steps, dt
    )
    assert state_traj.shape == (num_steps + 1, len(init_state))
    assert control_traj.shape == (num_steps, len(policy(init_state, 0)))

    state = init_state
    for step in range(num_steps):
        time = step * dt
        assert jnp.isclose(state_traj[step], state).all()
        control = policy(state, time)
        assert jnp.isclose(control_traj[step], control).all()
        state = test_dyn.discrete_dynamics(state, control, time, dt)

    assert jnp.isclose(state_traj[-1], state).all()
