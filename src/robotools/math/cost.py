"""Module for cost/loss/objective functions.

Many aspects use optimization (control, planning, estimation), and often there are
common things you'd like to do such as quadratically approximate the cost. This module
will also act as a library of common cost functions that can be easily referenced. Using
a class instead of having a quadratrization function will allow for the jacobian and
hessian functions to be calculated once and then applied to trajectory points.

Example usecase
    TV-LQR: Quadratrize the cost along the trajectory, solve and iterate
    SQP: Quadratrize the csot along the trajectory, solve and iterate

The library of cost functions will cast everything to numpy arrays to maintain
compatibility with cvxpy.
"""

import inspect
from typing import Callable, Optional, Union

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np  # for cvxpy compatibility


class Cost:
    """A discrete time cost function"""

    def __init__(
        self,
        stage_cost_fn: Optional[Callable] = None,
        terminal_cost_fn: Optional[Callable] = None,
    ):
        """
        This class can etiher be inherited and the cost functions overwritten, or
        functions can be passed in, but not both.

        Args:
            stage_cost_fn: the stage cost which takes in the state, control, and time
            terminal_cost_fn: the final cost which takes in the state
        """
        if stage_cost_fn is not None:
            self._check_signature(stage_cost_fn, 3, "Stage cost")
        else:
            stage_cost_fn = lambda state, control, time: 0
        self._stage = stage_cost_fn

        if terminal_cost_fn is not None:
            self._check_signature(terminal_cost_fn, 1, "Terminal cost")
        else:
            terminal_cost_fn = lambda state: 0
        self._terminal = terminal_cost_fn

        self.stage_grad = jax.grad(
            self.stage, (0, 1)
        )  # tuple of gradient arrays for state and control
        self.terminal_grad = jax.grad(self.terminal)  # just a gradient array

        self.stage_hessian = jax.hessian(self.stage, (0, 1))
        self.stage_hessian_cross = jax.grad(jax.grad(self.stage, 0), 1)
        self.terminal_hessian = jax.hessian(self.terminal)

    def _check_signature(self, fn, expected_args, name="This function"):
        sig = inspect.signature(fn)
        num_args = len(sig.parameters)
        if num_args != expected_args:
            raise ValueError(
                f"{name} must take exactly {expected_args} arguments, got {num_args}."
            )

    def stage(self, state, control, time):
        """The stage cost at a particular point in time"""
        return self._stage(state, control, time)

    def terminal(self, state):
        return self._terminal(state)

    def stage_quadratic(
        self,
        state: Union[jnp.ndarray, cp.Variable],
        control: Union[jnp.ndarray, cp.Variable],
        time: float,
        state_nom: Union[jnp.ndarray, cp.Parameter],
        control_nom: Union[jnp.ndarray, cp.Parameter],
    ):
        """Return the quadratic approximation of the cost given some nominal

        Args:
            state: a length n array
            control: a length m array
            time: the current time (in seconds)
            state_nom: the nominal state to approximate around
            control_nom: the nominal control to approximate around

        Returns the total cost. If state and control are cvxpy variables, returns a
        cvxpy expression. If state and control are jnp.ndarrays, returns a numeric value
        """
        nominal_cost = self.stage(state_nom)
        gradient_state, gradient_control = self.stage_grad(state_nom, control_nom, time)
        hessian_state, hessian_control = self.stage_hessian(
            state_nom, control_nom, time
        )
        hessian_cross = self.stage_hessian_cross(state_nom, control_nom, time)

        state_err = state - state_nom
        control_err = control - control_nom

        quadratic_cost = (
            nominal_cost
            + state_err @ gradient_state
            + control_err @ gradient_control
            + 0.5 * state_err @ (hessian_state @ state_err)
            + state_err @ (hessian_cross @ control_err)
            + 0.5 * control_err @ (hessian_control @ control_err)
        )
        return quadratic_cost

    def terminal_quadratic(
        self,
        state: Union[jnp.ndarray, cp.Variable],
        state_nom: Union[jnp.ndarray, cp.Parameter],
    ):
        """Return the quadratic approximation of the cost given some nominal

        Args:
            state: a length n array
            control: a length m array
            time: the current time (in seconds)
            state_nom: the nominal state to approximate around
            control_nom: the nominal control to approximate around

        Returns the total cost. If state and control are cvxpy variables, returns a
        cvxpy expression. If state and control are jnp.ndarrays, returns a numeric value
        """
        nominal_cost = self.terminal(state_nom)
        gradient_state = self.terminal_grad(state_nom)
        hessian_state = self.terminal_hessian(state_nom)

        quadratic_cost = (
            nominal_cost
            + (state - state_nom) @ gradient_state
            + 0.5 * (state - state_nom) @ hessian_state @ (state - state_nom)
        )
        return quadratic_cost

    def total(
        self,
        states: Union[jnp.ndarray, cp.Variable],
        controls: Union[jnp.ndarray, cp.Variable],
        dt: float,
    ) -> float:
        """Calculate or return an cvxpy expression for the total cost

        Args:
            states: a n x N+1 jnp array or cv variable
            controls: a m x N jnp array or cv variable
            dt: the time step
        """
        times = jnp.arange(states.shape[-1]) * dt

        running_cost = 0
        for state, control, time in zip(states[:-1], controls, times[:-1]):
            running_cost += self.stage(
                state, control, time
            )  # stage cost must operate with numpy operations to be compatible with cvxpy

        return running_cost + self.terminal(states[-1])

    def total_quadratic(self, states, controls, dt) -> float:
        times = jnp.arange(len(states)) * dt

        running_cost = jax.vmap(self.stage_quadratic, in_axes=[0, 0, 0])(
            states[:-1], controls, times
        )
        return running_cost + self.terminal_quadratic(states[-1])


def quadratic(Q: jnp.ndarray, R: jnp.ndarray, S: jnp.ndarray) -> tuple[Callable]:
    """Get quadratic stage and terminal cost functions
    Args:
        Q: an nxn state cost matrix
        R: an mxm control cost matrix
        S: an nxm cross cost matrix

    Returns a stage and terminal cost functinos that compute the quadratic cost
    """
    Q = np.array(Q)
    R = np.array(R)
    S = np.array(S)

    def stage_cost(state, control, time):
        return state @ Q @ state + control @ R @ control + state @ S @ control

    def terminal_cost(state):
        return state @ Q @ state

    return stage_cost, terminal_cost
