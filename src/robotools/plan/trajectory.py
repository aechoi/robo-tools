import os
from tkinter import filedialog
from typing import Self

import jax.numpy as jnp
import matplotlib.pyplot as plt


class Trajectory:
    """A discrete trajectory of states and controls."""

    def __init__(
        self, states: jnp.ndarray = None, controls: jnp.ndarray = None, dt: float = None
    ) -> None:
        if (
            (states is not None)
            and (controls is not None)
            and len(states) != len(controls) + 1
        ):
            raise ValueError("States should be one time step longer than the controls")

        self.states = states
        self.controls = controls
        self.dt = dt

    def save_traj(self, file_path: str = None) -> None:
        """Save the current trajectory as an npz file

        Args:
            file_path: relative or global file path
        """
        if file_path is None:
            file_path = _get_file_path_save()
        jnp.savez(
            file_path,
            states=jnp.array(self.states),
            controls=jnp.array(self.controls),
            dt=self.dt,
        )

    @classmethod
    def load_traj(cls, file_path: str = None) -> Self:
        """Generate a Trajectory object from a saved file

        Args:
            file_path: relative or global file path

        Returns a Trajectory object
        """
        if file_path is None:
            file_path = _get_file_path_open()
        loaded_data = jnp.load(file_path)
        return cls(loaded_data["states"], loaded_data["controls"], loaded_data["dt"])

    def plot(self) -> None:
        """Plot the trajectory states and controls"""
        _, ax = plt.subplots()
        c_ax = ax.twinx()

        time = jnp.arange(len(self.states)) * self.dt
        ax.plot(time, self.states, label=list(jnp.arange(len(self.states[0])) + 1))
        ax.legend(loc="upper left")

        c_ax.plot(
            time[:-1],
            self.controls,
            "--",
            label=list(jnp.arange(len(self.controls[0])) + 1),
        )
        c_ax.legend(loc="upper right")

        ax.set_xlabel("Time")
        ax.set_ylabel("States")
        c_ax.set_ylabel("Controls")
        plt.show()


def _get_file_path_open() -> str:
    """Graphically pick a file to open"""
    file_path = filedialog.askopenfilename(
        title="Select trajectory file",
        initialdir=os.path.dirname(os.path.abspath(__file__)),
        filetypes=[("Zipped Numpy", "*.npz")],
    )
    return file_path


def _get_file_path_save() -> str:
    """Graphically pick a path to save"""
    file_path = filedialog.asksaveasfilename(
        defaultextension=".npz",
        initialdir=os.path.dirname(os.path.abspath(__file__)),
        filetypes=[("Zipped Numpy", "*.npz"), ("All files", "*.*")],
        title="Save As",
    )
    return file_path
