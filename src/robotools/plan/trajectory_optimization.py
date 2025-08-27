from abc import ABC, abstractmethod

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np  # Used to populate cvxpy param values
from tqdm import tqdm

from robotools.plan.trajectory import Trajectory
from robotools.model.dynamics import DynamicModel
from robotools.math.cost import Cost
from robotools.math.constraint import Constraint

"""
Things defined by the problem
    loss function
    dynamics
    boundary conditions
    other constraints

Things determined by transcription method
    decision variables
    dynamic constraint approximation
    loss function approximation
    loss function quadratrization (wrt decision variables)
    constraint linearization (wrt decision variables)

Things determined by (and common) SQP
    act of solve, update, repeat until convergence
"""

"""TODO: do the technique of setting up the format of the cvxpy expression
with parameters, and calculate the grad with AD and populate.

Add other constraints
Handle time varying case
"""


class DirectMethod(ABC):
    """An abstract class meant to template the many methods of formulating
    trajectory optimization problems via direct methods.

    Direct methods transcribe the problem before solving. Transcription is the
    process by which a trajectory optimization problem is converted into a
    parameter optimization problem. This is sometimes referred to as
    discretization. Transcription methods generally fall into two categories:
    shooting methods and collocation methods."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def approximate(self, trajectory: Trajectory) -> cp.Problem:
        """Approximate the non-linear program as a convex quadratic program

        Generate a convex optimization problem by making a quadratic
        approximation of the objective and linear approximations of the
        constraints with respect to the decision variables

        Args:
            trajectory: the trajectory to approximate around

        Returns the approximated cvxpy problem
        """
        raise NotImplementedError()

    @abstractmethod
    def extract_trajectory(problem: cp.Problem) -> Trajectory:
        """Calculate the trajectory from the resulting problem solution.

        Args:
            problem: a solved cvxpy Problem

        Returns a trajectory object that results from the problem solution
        """
        raise NotImplementedError()


class DirectTranscription(DirectMethod):
    def __init__(
        self,
        num_steps: int,
        dt: float,
        cost: Cost,
        model: DynamicModel,
        constraints: list[Constraint] = None,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.cost = cost
        self.model = model

        if constraints is None:
            raise ValueError("Need at least an initial condition")
        self.constraints = constraints

        self.times = jnp.arange(num_steps + 1) * dt

    def extract_trajectory(problem):
        pass

    def construct_nlp(self):
        # Decision Variables
        state_traj = cp.Variable((self.num_steps + 1, self.model.n))
        control_traj = cp.Variable((self.num_steps, self.model.m))

        # Objective
        objective = cp.Minimize(self.cost.total(state_traj, control_traj, self.dt))

        # Constraints
        constraints = []
        constraints += (
            self.model.discrete_dynamics(
                state_traj[:-1].T, control_traj.T, self.times, self.dt
            )
            == state_traj[1:].T
        )
        # TODO create constraint class
        constraints += state_traj[:, 0] == jnp.array([1.0, 1.0])

        problem = cp.Problem(objective, constraints)
        return problem

    def approximate(self, trajectory: Trajectory):
        """
        Quadratrize cost around traj, linearize constraints around traj
        Assign values to prestructured cvxpy parameters
        Return problem
        """
        num_steps = len(trajectory.controls)
        times = jnp.arange(num_steps + 1) * trajectory.dt

        As = cp.Parameter((num_steps, self.model.n, self.model.n))
        Bs = cp.Parameter((num_steps, self.model.n, self.model.m))
        Cs = cp.Parameter((num_steps, self.model.n))

        # Linearize dynamics
        A_values, B_values, C_values = self.model.discrete_linearize_traj(
            trajectory.states, trajectory.controls, times, trajectory.dt
        )
        As.value = np.array(A_values)
        Bs.value = np.array(B_values)
        Cs.value = np.array(C_values)


def SQP(
    direct_method: DirectMethod,
    init_trajectory: Trajectory,
    max_sqp_iterations: int = 100,
    convergence_tol: float = 1e-3,
) -> Trajectory:
    current_trajectory = init_trajectory
    prev_cost = jnp.inf

    for _ in tqdm(range(max_sqp_iterations)):
        problem = direct_method.approximate(current_trajectory)
        result = problem.solve()

        tqdm.write(f"Cost: {result}")

        current_trajectory = direct_method.extract_trajectory(problem)

        if jnp.abs(prev_cost - result) / result < convergence_tol:
            break
        prev_cost = result

    return current_trajectory


def cost_deviation(
    states: cp.Variable,
    controls: cp.Variable,
    prev_states: cp.Parameter,
    prev_controls: cp.Parameter,
    coeff: float = 0.01,
) -> cp.Expression:
    """Calculate the cost of states and controls deviating between trajectories
    when updating via SQP

    Args:
        states: a N+1 x n array of states
        controls: a N x m array of states
        prev_states: a N+1 x n array of states
        prev_controls: a N x m array of states
        coeff: the amount of weight to apply to the sum of squares"""
    cost = 0
    cost += coeff * cp.sum_squares(states - prev_states)
    cost += coeff * cp.sum_squares(controls - prev_controls)
    return cost


if __name__ == "__main__":
    from robotools.model.library import DoubleIntegrator
    from robotools.math import cost

    num_steps = 100
    dt = 0.1

    model = DoubleIntegrator()
    # cost
    Q = jnp.eye(model.n)
    R = jnp.eye(model.m)
    S = jnp.zeros((model.n, model.m))
    stage_cost, terminal_cost = cost.quadratic(Q, R, S)
    cost_expression = Cost(stage_cost, terminal_cost)

    constraints = []

    direct_transcription = DirectTranscription(
        num_steps, dt, cost_expression, model, constraints
    )

    prob = direct_transcription.construct_nlp()
    prob.solve()


# def seq_quad_prog(
#     model,
#     cost,
#     num_samples,
#     dt,
#     init_state,
#     final_state,
#     final_control,
#     num_iter=100,
#     init_traj: Trajectory = None,
#     constraint_func=None,
# ):
#     """An implementation of sequential quadratic programming"""

#     ### Initial Solve

#     ## Define variables and params
#     xs = cp.Variable((num_samples, model.n))
#     us = cp.Variable((num_samples, model.m))
#     xs_prev = cp.Parameter((num_samples, model.n))
#     us_prev = cp.Parameter((num_samples, model.m))

#     As = [cp.Parameter((model.n, model.n)) for _ in range(num_samples - 1)]
#     Bs = [cp.Parameter((model.n, model.m)) for _ in range(num_samples - 1)]
#     Cs = [cp.Parameter(model.n) for _ in range(num_samples - 1)]

#     ## Define constraints
#     if constraint_func is None:
#         constraints = []
#     else:
#         constraints = constraint_func(xs, us)
#     constraints += [xs[0] == np.array(init_state)]
#     constraints += [xs[-1] == np.array(final_state)]
#     constraints += [us[-1] == np.array(final_control)]
#     for t in range(num_samples - 1):
#         constraints += [
#             xs[t + 1] == As[t] @ xs[t] + Bs[t] @ us[t] + Cs[t]
#         ]  # dynamics constraint

#     ## Define objective
#     objective = cost(xs, us)
#     objective = objective + cost_deviation(xs, us, xs_prev, us_prev)

#     prob = cp.Problem(cp.Minimize(objective), constraints)

#     ## Set problem params
#     if init_traj is None:
#         prev_controls = jnp.zeros([num_samples, model.m])
#         # prev_states, _ = model.simulate(
#         #     init_state, get_open_loop_policy(prev_controls[:-1]), num_samples, dt
#         # )
#         prev_states = jnp.linspace(init_state, final_state, num_samples)
#     else:
#         prev_states = init_traj.states
#         prev_controls = init_traj.controls
#     xs_prev.value = np.array(prev_states)
#     us_prev.value = np.array(prev_controls)

#     prev_cost = jnp.inf

#     for _ in tqdm(range(num_iter)):
#         A_values, B_values, C_values = model.d_linearize_traj(
#             prev_states, prev_controls[:-1], 0, dt
#         )
#         for t in range(num_samples - 1):
#             As[t].value = np.array(A_values[t])
#             Bs[t].value = np.array(B_values[t])
#             Cs[t].value = np.array(C_values[t])

#         result = prob.solve()
#         tqdm.write(f"Cost: {result}")

#         # If dynamics are highly nonlinear, can be unstable
#         prev_controls = us.value
#         prev_states = xs.value
#         # if jnp.abs(prev_cost - result) / result > 0.01:
#         #     prev_states = xs.value
#         # else:
#         #     prev_states, _ = model.simulate(
#         #         init_state, get_open_loop_policy(jnp.asarray(prev_controls)), num_samples, dt
#         #     )
#         if jnp.abs(prev_cost - result) / result < 0.0001:
#             break
#         prev_cost = result

#         xs_prev.value = np.array(prev_states)
#         us_prev.value = np.array(prev_controls)
#     print("Trajed init state", prev_states[0])
#     return prev_states, prev_controls


# def get_open_loop_policy(controls, times):
#     """Given a sequence of controls, return a policy function assuming zoh

#     Args:
#         controls: an Nxm array of controls
#         times: an N+1 array of timestamps

#     Returns a function which provides control input given a time (float)

#     TODO: maybe move this to a different module
#     """

#     def open_loop_policy(states, t):
#         return controls[jnp.searchsorted(times[:-1], t, "right") - 1]

#     return open_loop_policy
