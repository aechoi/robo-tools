from abc import ABC, abstractmethod
from typing import Callable

from jax import vmap, jacobian, lax
import jax.numpy as jnp

from robotools.math import integrate


class DynamicModel(ABC):
    def __init__(self):
        self.n = None
        self.m = None

        self.ct_jac_state = jacobian(self.continuous_dynamics, 0)
        self.ct_jac_control = jacobian(self.continuous_dynamics, 1)
        self.dt_jac_state = jacobian(self.discrete_dynamics, 0)
        self.dt_jac_control = jacobian(self.discrete_dynamics, 1)

    @abstractmethod
    def continuous_dynamics(
        self, state: jnp.ndarray, control: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """Continuous dynamics that return the derivative of the state.

        Args:
            state: a length n jnp.array
            control: a length m jnp.array
            t: the current time

        Returns the derivative of the state
        """
        raise NotImplementedError

    def discrete_dynamics(
        self, state: jnp.ndarray, control: jnp.ndarray, t: float, dt: float
    ) -> jnp.ndarray:
        """Discrete dynamics that return the next state. Overwrite with other
        integrators as desired.

        Args:
            state: a length n jnp.array
            control: a length m jnp.array
            t: the current time
            dt: the time step between each discrete step

        Returns the next state
        """

        # integration methods expect a certain argument order which does not
        # match the canonical form of the dynamics arguments
        def dynamics(t, state, control):
            return self.continuous_dynamics(state, control, t)

        return integrate.rk4(dynamics, dt, t, state, control)

    def discrete_simulate(
        self,
        init_state: jnp.ndarray,
        policy: Callable,
        num_steps: int,
        dt: float,
        init_time: float = 0,
    ) -> tuple[jnp.ndarray]:
        """Simulate a policy from some initial condition.

        Args:
            init_state: a length n array of the initial state
            policy: a policy that takes the arguments (state, t) to determine a
                length m jnp.ndarray of controls
            num_steps: the number of time steps to simulate. In other words,
                time spans from 0 to num_steps*dt. The state at time=0 is the
                initial state, and num_steps states are calculated to make a
                total of num_steps+1 states in the trajectory.
            dt: the time step between each discrete step

        Returns a num_steps+1 x n jnp.array of states and a num_steps x m
        jnp.array of controls. The first state is init_state.
        """

        def step_fn(carry, time):
            state = carry
            control = policy(state, time)
            next_state = self.discrete_dynamics(state, control, time, dt)
            return next_state, (next_state, control)

        time_steps = jnp.arange(init_time, init_time + num_steps * dt, dt)
        _, (states, controls) = lax.scan(
            step_fn,
            init_state,
            time_steps,
        )

        states = jnp.vstack([init_state[None, :], states])

        return states, controls

    def continuous_linearize(
        self, state: jnp.ndarray, control: jnp.ndarray, t: float
    ) -> tuple[jnp.ndarray]:
        """Linearize the continuous dynamics about the provided state and control

        Args:
            state: a length n state array to linearize about
            control: a length m control array to linearize about
            t: the current time

        Returns the terms for the linearized dxdt = Ax + Bu + C
        """
        A = self.ct_jac_state(state, control, t)
        B = self.ct_jac_control(state, control, t)
        C = self.continuous_dynamics(state, control, t) - A @ state - B @ control

        return A, B, C

    def continuous_linearize_traj(
        self, states: jnp.ndarray, controls: jnp.ndarray, ts: jnp.ndarray
    ) -> tuple[jnp.ndarray]:
        """Linearize the continuous dynamics along some trajectory

        Args:
            states: an N+1xn array where N+1 is the length of the trajectory
                including the initial state and the final state
            controls: an Nxm array of controls
            ts: a length N+1 array of time points of the trajectory to
                linearize around

        Returns the linearized matrices and vectors along the entire
        trajectory such that dxdt[i] = A_s[i] x[i] + B_s[i] + C_s[i]
        """
        A_s, B_s, C_s = vmap(self.continuous_linearize, in_axes=[0, 0, 0])(
            states[:-1], controls, ts[:-1]
        )
        return A_s, B_s, C_s

    def discrete_linearize(
        self, state: jnp.ndarray, control: jnp.ndarray, t: float, dt: float
    ) -> tuple[jnp.ndarray]:
        """Linearize the discrete dynamics about the reference point

        Args:
            state: a length n state array to linearize about
            control: a length m control array to linearize about
            t: the current time
            dt: the time step

        Returns the terms for the linearized x[i+1] = Ax + Bu + C"""
        A = self.dt_jac_state(state, control, t, dt)
        B = self.dt_jac_control(state, control, t, dt)
        C = self.discrete_dynamics(state, control, t, dt) - A @ state - B @ control

        return A, B, C

    def discrete_linearize_traj(
        self, states: jnp.ndarray, controls: jnp.ndarray, t: jnp.ndarray, dt: float
    ) -> tuple[jnp.ndarray]:
        """Linearize the discrete dynamics along some trajectory

        Args:
            states: an N+1xn array where N+1 is the length of the trajectory
                including the initial state and the final state
            controls: an Nxm array of controls
            ts: a length N+1 array of time points of the trajectory to
                linearize around
            dt: the time step

        Returns the linearized matrices and vectors along the entire
        trajectory such that x[i+1] = A_s[i] x[i] + B_s[i] + C_s[i]
        """
        A_s, B_s, C_s = vmap(self.discrete_linearize, in_axes=[0, 0, 0, None])(
            states[:-1], controls, t[:-1], dt
        )
        return A_s, B_s, C_s
