"""Implementation of various numeric integration methods"""

from typing import Callable

import jax.numpy as jnp


def euler(
    dynamics: Callable, dt: float, t: float, var: jnp.ndarray, *args
) -> jnp.ndarray:
    """Euler integration

    Args:
        dynamics: a function that returns the derivative of var. Has arguments
            in the order of t, var, *args
        dt: the step in time
        t: the current time
        var: the variable whose derivative is being taken
        args: arguments for the dynamics

    Returns the approximate value after dt amount of time
    """
    return var + dt * dynamics(t, var, *args)


def rk4(
    dynamics: Callable, dt: float, t: float, var: jnp.ndarray, *args
) -> jnp.ndarray:
    """RK4 integration

    Args:
        dynamics: a function that returns the derivative of var. Has arguments
            in the order of t, var, *args
        dt: the step in time
        t: the current time
        var: the variable whose derivative is being taken
        args: arguments for the dynamics

    Returns the approximate value after dt amount of time
    """

    k1 = dynamics(t, var, *args)
    k2 = dynamics(t + dt / 2, var + k1 / 2 * dt, *args)
    k3 = dynamics(t + dt / 2, var + k2 / 2 * dt, *args)
    k4 = dynamics(t + dt, var + k3 * dt, *args)
    return var + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
