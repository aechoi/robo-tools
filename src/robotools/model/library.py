import jax.numpy as jnp
from robotools.model.dynamics import DynamicModel


class DoubleIntegrator(DynamicModel):
    """Simple 1-dimensional double integrator"""

    def __init__(self, mass=1):
        super().__init__()
        self.mass = mass

        self.n = 2
        self.m = 1

    def continuous_dynamics(self, state, control, t):
        pos, vel = state
        return jnp.array([vel, control[0] / self.mass])


class Pendulum(DynamicModel):
    """Simple pendulum, theta is 0 down"""

    def __init__(self, l=0.5, mass=1):
        super().__init__()
        self.g = 9.8
        self.l = l
        self.b = 0.1
        self.mass = mass

        self.n = 2
        self.m = 1

    def continuous_dynamics(self, state, control, t):
        theta, dtheta = state
        f = jnp.array(
            [
                dtheta,
                -self.g / self.l * jnp.sin(theta)
                - self.b / (self.mass * self.l**2) * dtheta,
            ]
        )

        g_aff = jnp.array([0, 1 / (self.mass * self.l**2)])
        return f + g_aff * control


class PlanarQuad(DynamicModel):
    def __init__(self, mass=1, I=1, r=1):
        super().__init__()
        self.g = 9.81
        self.mass = mass
        self.I = I
        self.r = r

        self.n = 6
        self.m = 2

    def continuous_dynamics(self, state, control, t):
        x, dx, y, dy, theta, dtheta = state
        u1, u2 = control
        dynamics = jnp.array(
            [
                dx,
                -(u1 + u2) / self.mass * jnp.sin(theta),
                dy,
                (u1 + u2) * jnp.cos(theta) / self.mass - self.g,
                dtheta,
                self.r / self.I * (u1 - u2),
            ]
        )
        return dynamics


class CartPole(DynamicModel):
    def __init__(self):
        super().__init__()
        self.g = 9.81
        self.l = 1
        self.m_p = 1
        self.m_c = 1

        self.n = 4
        self.m = 1

    def continuous_dynamics(self, state, control, t):
        x, dx, theta, dtheta = state
        mass_eff = 1 / (self.m_c + self.m_p * jnp.sin(theta) ** 2)
        f = jnp.array(
            [
                dx,
                mass_eff
                * (
                    control[0]
                    + self.m_p
                    * jnp.sin(theta)
                    * (self.l * dtheta**2 + self.g * jnp.cos(theta))
                ),
                dtheta,
                mass_eff
                / self.l
                * (
                    -control[0] * jnp.cos(theta)
                    - self.m_p * self.l * dtheta**2 * jnp.cos(theta) * jnp.sin(theta)
                    - (self.m_c + self.m_p) * self.g * jnp.sin(theta)
                ),
            ]
        )
        return f
