import os

import jax.random as rnd

from robotools.plan import trajectory


def test_save_load():
    num_states = 2
    num_controls = 1
    num_steps = 10
    dt = 0.1

    key = rnd.key(0)
    states = rnd.uniform(key, (num_steps + 1, num_states))
    controls = rnd.uniform(key, (num_steps, num_controls))

    traj = trajectory.Trajectory(states, controls, dt)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = "test_file.npz"
    path = os.path.join(dir_path, file_path)

    traj.save_traj(path)

    loaded_traj = trajectory.Trajectory.load_traj(path)

    assert (traj.states == loaded_traj.states).all()
    assert (traj.controls == loaded_traj.controls).all()
    assert traj.dt == loaded_traj.dt

    os.remove(path)
