import ast
from dataclasses import dataclass, field, replace
from typing import Union

import cvxpy as cvx
import numpy as np
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)

from pdm4ar.exercises.ex13.discretization import *
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams
from pdm4ar.exercises.ex13.satellite import SatelliteDyn
from pdm4ar.exercises_def.ex13.goal import DockingTarget


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "CLARABEL"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 25  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 350.0  # slack variable weight
    lambda_obs: float = 950.0  # obstacle slack weight
    weight_p: NDArray = field(default_factory=lambda: 0.12 * np.array([[1.0]]).reshape((1, -1)))

    tr_radius: float = 14.0  # initial trust region radius
    min_tr_radius: float = 0.2  # min trust region radius
    max_tr_radius: float = 50  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.2  # div factor trust region update
    beta: float = 3.0  # mult factor trust region update

    # Separate trust region radii (per element, inf-norm)
    tr_radius_x: float = 3.0
    tr_radius_u: float = 1.0
    tr_radius_p: float = 2.0
    tr_radius_step_x: float = 2.5
    tr_radius_step_u: float = 0.8

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-3  # Stopping criteria constant


class SatellitePlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    satellite: SatelliteDyn
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
        docking_goal: DockingTarget | None = None,
        boundary_box: tuple[float, float, float, float] | None = None,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp
        # (xmin, xmax, ymin, ymax) defining the playable area if provided by the scenario
        self.boundary_box = boundary_box

        # Solver Parameters
        self.params = SolverParameters()

        # Satellite Dynamics
        self.satellite = SatelliteDyn(self.sg, self.sp)
        self._setup_scaling()

        # Discretization Method
        self.integrator = FirstOrderHold(self.satellite, self.params.K, self.params.N_sub)

        # Check dynamics implementation
        if not self.integrator.check_dynamics():
            raise ValueError("Dynamics check failed.")
        else:
            print("Dynamics check passed.")

        # Static circular obstacles (planets)
        self.n_planets = len(self.planets)
        self.safe_margin = 2.1
        self.planet_names = list(self.planets.keys())
        print(f"Planner sees {self.n_planets} planets: {self.planet_names}")

        # Dynamic circular obstacles (asteroids)
        self.n_asteroids = len(self.asteroids)
        self.asteroid_names = list(self.asteroids.keys())
        print(f"Planner sees {self.n_asteroids} asteroids: {self.asteroid_names}")
        if self.n_asteroids > 0 and self.params.solver.upper() == "CLARABEL":
            print("Switching SCvx solver to ECOS for better robustness with moving obstacles.")
            self.params = replace(self.params, solver="ECOS")

        # Variables and parameters of the optimisation problem
        self.variables = self._get_variables()
        self.problem_parameters = self._get_problem_parameters()
        self.docking_halfspaces: list[tuple[np.ndarray, float]] = []
        self.dock_start_idx: int = 0
        self.docking_anchor: np.ndarray | None = None
        self.goal_tolerances: dict[str, float] | None = None
        if docking_goal is not None:
            self._setup_docking_halfspaces(docking_goal)
            self._update_goal_tolerances(docking_goal)

        constraints = self._get_constraints()
        objective = self._get_objective()

        # Cvx optimisation problem
        self.problem = cvx.Problem(objective, constraints)

    def _setup_scaling(self):
        """
        Define simple constant scaling so that all optimisation quantities
        (states, inputs and final time) are roughly within [0, 1].
        """

        # workspace is roughly a 20x20 square centred at the origin
        pos_scale = 9.0
        # allow yaw to span [-pi, pi]
        psi_scale = float(np.pi)
        # coarse guesses for achievable velocities/angular rate
        vel_scale = 3.0
        dpsi_scale = 1.8

        F_min, F_max = self.sp.F_limits
        max_force = max(abs(F_min), abs(F_max), 1.0)

        self.scale_x = np.array([pos_scale, pos_scale, psi_scale, vel_scale, vel_scale, dpsi_scale])
        self.scale_u = np.array([max_force, max_force])
        # optimisation bounds limit final time; keep scale close to expected tf
        self.scale_p = 12.0

        self.inv_scale_x = 1.0 / self.scale_x
        self.inv_scale_u = 1.0 / self.scale_u
        self.inv_scale_p = 1.0 / self.scale_p

        self.scale_mat_x = np.diag(self.inv_scale_x)
        self.scale_mat_u = np.diag(self.inv_scale_u)

    def _update_goal_tolerances(self, docking_goal: DockingTarget):
        self.goal_tolerances = {
            "pos": float(docking_goal.pos_tol),
            "dir": float(docking_goal.dir_tol),
            "vel": float(docking_goal.vel_tol),
        }

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState, docking_goal: DockingTarget | None = None
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:

        self.init_state = init_state
        self.goal_state = goal_state
        # refresh docking constraints if provided, otherwise keep previous config (e.g. replanning)
        if docking_goal is not None:
            self._setup_docking_halfspaces(docking_goal)
            self._update_goal_tolerances(docking_goal)

        # initial guess
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        X_curr = self.X_bar
        U_curr = self.U_bar
        p_curr = float(self.p_bar[0])
        self.curr_tr_x = self.params.tr_radius_x
        self.curr_tr_u = self.params.tr_radius_u
        self.curr_tr_p = self.params.tr_radius_p

        for it in range(self.params.max_iterations):
            self.X_bar = X_curr
            self.U_bar = U_curr
            self.p_bar = np.array([p_curr])

            # met à jour les paramètres (A_k, B_k, X_ref, …)
            self._convexification()

            try:
                self.problem.solve(
                    verbose=self.params.verbose_solver,
                    solver=self.params.solver,
                )
            except cvx.SolverError:
                print(f"❌ Solver failed at iter {it}. Returning last feasible trajectory.")
                break

            status = self.problem.status
            has_solution = (
                self.variables["X"].value is not None
                and self.variables["U"].value is not None
                and self.variables["p"].value is not None
            )

            if not has_solution:
                print(f"❌ No candidate solution at iter {it} (status = {status}).")
                break

            X_new = self.variables["X"].value
            U_new = self.variables["U"].value
            p_new = float(self.variables["p"].value)

            scaled_step_X = (X_new - X_curr) * self.inv_scale_x[:, None]
            scaled_step_U = (U_new - U_curr) * self.inv_scale_u[:, None]
            step_X = np.linalg.norm(scaled_step_X, ord=np.inf)
            step_U = np.linalg.norm(scaled_step_U, ord=np.inf)
            step_p = abs(p_new - p_curr) * float(self.inv_scale_p)

            # si le problème est optimal (ou presque), on peut continuer les itérations
            if status in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
                step = max(step_X, step_U, step_p)

                X_curr, U_curr, p_curr = X_new, U_new, p_new

                self._update_trust_region(step_X, step_U, step_p, success=True)
                if self._check_convergence(step):
                    break

            else:
                # statut "mauvais" (infeasible_inaccurate, infeasible, …)
                X_curr, U_curr, p_curr = X_new, U_new, p_new
                self._update_trust_region(step_X, step_U, step_p, success=False)
                print(f"❌ Problem not optimal at iter {it} (status = {status}). Continuing with trust region shrink.")
                continue

        # ---------- LOG propre basé sur X_curr, U_curr, p_curr ----------
        print(f"[SCvx] Final p = {p_curr:.3f}")
        print(f"[SCvx] Mean thrust left  = {np.mean(U_curr[0, :]):.3f}")
        print(f"[SCvx] Mean thrust right = {np.mean(U_curr[1, :]):.3f}")

        x_final = X_curr[:, -1]
        goal_vec = np.array(
            [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                self.goal_state.dpsi,
            ]
        )

        pos_err = np.linalg.norm(x_final[0:2] - goal_vec[0:2])
        psi_err = abs(x_final[2] - goal_vec[2])
        vel_err = np.linalg.norm(x_final[3:5] - goal_vec[3:5])

        print(f"[SCvx] |X_final - goal|  = {np.linalg.norm(x_final - goal_vec):.3f}")
        print(f"[SCvx] |pos error|       = {pos_err:.3f}")
        print(f"[SCvx] psi error (rad)   = {psi_err:.3f}")
        print(f"[SCvx] |vel error|       = {vel_err:.3f}")

        # conversion en DgSampledSequence
        return self._convert_to_sequences(X_curr, U_curr, p_curr)

    def _get_frame_limits(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        """
        Return the environment bounds, optionally shrunk by `margin` on each side.
        """
        if self.boundary_box is not None:
            xmin, xmax, ymin, ymax = self.boundary_box
        else:
            # fallback to a generous default if no boundary is specified by the scenario
            xmin, xmax, ymin, ymax = -11.0, 11.0, -11.0, 11.0

        return xmin + margin, xmax - margin, ymin + margin, ymax - margin

    def _convert_to_sequences(
        self, X: NDArray, U: NDArray, p: float
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:

        K = self.params.K
        n_x = self.satellite.n_x

        # sécurité : éviter un horizon de temps quasi nul
        if p <= 0.1:
            p = 0.1

        # timestamps: uniform in [0, p]
        ts = np.linspace(0.0, p, K)

        # Build command sequence
        cmds_list = [SatelliteCommands(F_left=float(U[0, k]), F_right=float(U[1, k])) for k in range(K)]
        cmds_seq = DgSampledSequence[SatelliteCommands](timestamps=ts.tolist(), values=cmds_list)

        # Build state sequence
        states_list = [
            SatelliteState(
                x=float(X[0, k]),
                y=float(X[1, k]),
                psi=float(X[2, k]),
                vx=float(X[3, k]),
                vy=float(X[4, k]),
                dpsi=float(X[5, k]),
            )
            for k in range(K)
        ]
        states_seq = DgSampledSequence[SatelliteState](timestamps=ts.tolist(), values=states_list)

        return cmds_seq, states_seq

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Build a simple but meaningful initial guess trajectory (X_bar, U_bar, p_bar).

        Robust generic version:
        - Collect all circular obstacles (planets + asteroid initial positions).
        - Build two candidate corridors: one BELOW all obstacles, one ABOVE all.
        - Keep only corridors that stay inside the walls.
        - Choose the cheaper corridor (less vertical deviation from start & goal).
        - If no corridor is valid, fall back to straight line.
        """

        K = self.params.K
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        # containers
        X = np.zeros((n_x, K))
        U = np.zeros((n_u, K))
        p = np.zeros((n_p,))

        # ----- Extract initial and goal states -----
        x0 = np.array(
            [
                self.init_state.x,
                self.init_state.y,
                self.init_state.psi,
                self.init_state.vx,
                self.init_state.vy,
                self.init_state.dpsi,
            ]
        )

        xg = np.array(
            [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                self.goal_state.dpsi,
            ]
        )

        tau = np.linspace(0.0, 1.0, K)

        # ----- Collect circular obstacles (planets + asteroid initial positions) -----
        obstacle_centers: list[np.ndarray] = []
        obstacle_radii: list[float] = []

        for pname in self.planet_names:
            pl = self.planets[pname]
            obstacle_centers.append(np.array(pl.center, dtype=float))
            obstacle_radii.append(float(pl.radius))

        for aname in getattr(self, "asteroid_names", []):
            ast = self.asteroids[aname]
            obstacle_centers.append(np.array(ast.start, dtype=float))
            obstacle_radii.append(float(ast.radius))

        # Frame (same as in constraints)
        margin = 0.8
        x_min, x_max, y_min, y_max = self._get_frame_limits(margin)

        segments: list[np.ndarray]

        if len(obstacle_centers) == 0:
            segments = [x0[:2], xg[:2]]
        else:
            y_low_raw = min(c[1] - (r + self.safe_margin + 1.0) for c, r in zip(obstacle_centers, obstacle_radii))
            y_high_raw = max(c[1] + (r + self.safe_margin + 1.0) for c, r in zip(obstacle_centers, obstacle_radii))

            y_low = float(np.clip(y_low_raw, y_min, y_max))
            y_high = float(np.clip(y_high_raw, y_min, y_max))

            corridor_low_valid = y_low >= y_min + 1e-6
            corridor_high_valid = y_high <= y_max - 1e-6

            cost_low = abs(y_low - x0[1]) + abs(y_low - xg[1]) if corridor_low_valid else np.inf
            cost_high = abs(y_high - x0[1]) + abs(y_high - xg[1]) if corridor_high_valid else np.inf

            if cost_low == np.inf and cost_high == np.inf:
                segments = [x0[:2], xg[:2]]
            else:
                y_corr = y_low if cost_low <= cost_high else y_high
                mid_x = 0.5 * (x0[0] + xg[0])
                segments = [
                    x0[:2],
                    np.array([x0[0], y_corr]),
                    np.array([mid_x, y_corr]),
                    xg[:2],
                ]

        # Bias initial motion eastwards to clear left-side traffic
        x_push = min(x0[0] + 6.0, xg[0] - 1.0, x_max)
        if x_push > segments[0][0] + 0.5:
            segments.insert(1, np.array([x_push, segments[0][1]]))

        def interp_segment(p_start: np.ndarray, p_end: np.ndarray, s: float) -> np.ndarray:
            return (1.0 - s) * p_start + s * p_end

        # ----- Insert docking anchor just before the goal (keeps guess inside funnel) -----
        if self.docking_anchor is not None and np.linalg.norm(segments[-1] - self.docking_anchor) > 1e-3:
            segments.insert(len(segments) - 1, self.docking_anchor)

        path_length = 0.0
        for i in range(len(segments) - 1):
            path_length += float(np.linalg.norm(segments[i + 1] - segments[i]))
        path_length = max(path_length, float(np.linalg.norm(xg[:2] - x0[:2])))
        avg_speed = 2.8
        buffer = 3.0 + (1.0 if self.docking_anchor is not None else 0.0)
        t_f_guess = path_length / max(avg_speed, 0.2) + buffer
        if self.docking_anchor is not None:
            t_f_guess *= 1.15
        p[:] = float(np.clip(t_f_guess, 5.0, 12.0))

        # ----- Piecewise-linear interpolation along segments -----
        if len(segments) == 2:
            # simple case: one segment
            for k, t in enumerate(tau):
                pos = interp_segment(segments[0], segments[1], t)
                X[0, k] = pos[0]
                X[1, k] = pos[1]
        else:
            # three segments: [0]->[1], [1]->[2], [2]->[3]
            for k, t in enumerate(tau):
                if t <= 1.0 / 3.0:
                    s = t / (1.0 / 3.0)
                    pos = interp_segment(segments[0], segments[1], s)
                elif t <= 2.0 / 3.0:
                    s = (t - 1.0 / 3.0) / (1.0 / 3.0)
                    pos = interp_segment(segments[1], segments[2], s)
                else:
                    s = (t - 2.0 / 3.0) / (1.0 / 3.0)
                    pos = interp_segment(segments[-2], segments[-1], s)

                X[0, k] = pos[0]
                X[1, k] = pos[1]

        # ----- Orientation guess: keep initial heading -----
        X[2, :] = x0[2]

        # ----- Velocity guess: finite-difference of positions over time -----
        dt_guess = t_f_guess / max(K - 1, 1)
        vx = np.gradient(X[0, :], dt_guess)
        vy = np.gradient(X[1, :], dt_guess)
        X[3, :] = vx
        X[4, :] = vy

        # ----- Angular velocity guess -----
        X[5, :] = 0.0

        # ----- Input guess (mild forward thrust, slightly asymmetric) -----
        thrust = 0.5
        U[0, :] = thrust * 0.8
        U[1, :] = thrust * 1.2

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        K = self.params.K

        variables = {
            "X": cvx.Variable((n_x, K)),
            "U": cvx.Variable((n_u, K)),
            "p": cvx.Variable(self.satellite.n_p),
            # virtual control for SCvx (keeps problem feasible)
            "nu": cvx.Variable((n_x, K - 1)),
        }

        if self.n_planets > 0:
            variables["s_planets"] = cvx.Variable((self.n_planets, K))
        if self.n_asteroids > 0:
            variables["s_asteroids"] = cvx.Variable((self.n_asteroids, K))

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx (dynamics + references + obstacle
        linearization). These are filled at each convexification step.
        """
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p
        K = self.params.K

        problem_parameters: dict[str, cvx.Parameter] = {
            # initial and goal state (6D : x, y, psi, vx, vy, dpsi)
            "init_state": cvx.Parameter(n_x),
            "goal_state": cvx.Parameter(n_x),
            # Discrete-time linear dynamics matrices (FOH), flattened per time step
            "A_bar": cvx.Parameter((n_x * n_x, K - 1)),  # each column: vec(A_k)
            "B_plus_bar": cvx.Parameter((n_x * n_u, K - 1)),  # each column: vec(B_plus_k)
            "B_minus_bar": cvx.Parameter((n_x * n_u, K - 1)),  # each column: vec(B_minus_k)
            "F_bar": cvx.Parameter((n_x * n_p, K - 1)),  # each column: vec(F_k)
            "r_bar": cvx.Parameter((n_x, K - 1)),  # each column: r_k
            # Reference trajectory for trust region / logging
            "X_ref": cvx.Parameter((n_x, K)),
            "U_ref": cvx.Parameter((n_u, K)),
            "p_ref": cvx.Parameter(n_p),
            # trust region radii (inf-norm)
            "tr_radius_x": cvx.Parameter(nonneg=True),
            "tr_radius_u": cvx.Parameter(nonneg=True),
            "tr_radius_p": cvx.Parameter(nonneg=True),
        }

        # Obstacle params for planets, linearised at each step
        if self.n_planets > 0:
            for i in range(self.n_planets):
                problem_parameters[f"obs_A_{i}"] = cvx.Parameter((2, K))  # column k = a_k
                problem_parameters[f"obs_b_{i}"] = cvx.Parameter(K)  # entry k = b_k

        # Obstacle params for dynamic asteroids (time-varying centers)
        if self.n_asteroids > 0:
            for i in range(self.n_asteroids):
                problem_parameters[f"ast_A_{i}"] = cvx.Parameter((2, K))  # column k = a_k
                problem_parameters[f"ast_b_{i}"] = cvx.Parameter(K)  # entry k = b_k

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]
        nu = self.variables["nu"]
        s_planets = self.variables.get("s_planets", None)
        s_asteroids = self.variables.get("s_asteroids", None)

        pp = self.problem_parameters

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p
        K = self.params.K

        constraints: list[cvx.Constraint] = []

        # --- final time within bounds ---
        constraints.append(p >= 0)
        constraints.append(p <= 14.0)
        if s_planets is not None:
            constraints.append(s_planets >= 0)
        if s_asteroids is not None:
            constraints.append(s_asteroids >= 0)

        # --- initial state ---
        constraints.append(X[:, 0] == pp["init_state"])

        constraints.append(cvx.norm(U[:, 0], 2) <= 0.1)

        A_bar = pp["A_bar"]
        B_plus_bar = pp["B_plus_bar"]
        B_minus_bar = pp["B_minus_bar"]
        F_bar = pp["F_bar"]
        r_bar = pp["r_bar"]

        p_vec = cvx.reshape(p, (n_p, 1))

        # ---------- Linearised dynamics (FOH) + slack nu ----------
        for k in range(K - 1):
            A_k = cvx.reshape(A_bar[:, k], (n_x, n_x), order="F")
            B_plus_k = cvx.reshape(B_plus_bar[:, k], (n_x, n_u), order="F")
            B_minus_k = cvx.reshape(B_minus_bar[:, k], (n_x, n_u), order="F")
            F_k = cvx.reshape(F_bar[:, k], (n_x, n_p), order="F")
            r_k = r_bar[:, k]

            Fp_k = cvx.reshape(F_k @ p_vec, (n_x,))

            constraints.append(
                X[:, k + 1] == A_k @ X[:, k] + B_minus_k @ U[:, k] + B_plus_k @ U[:, k + 1] + Fp_k + r_k + nu[:, k]
            )

        # ---------- Terminal constraints ----------
        goal = pp["goal_state"]
        pos_tol = self.goal_tolerances["pos"] if self.goal_tolerances is not None else 0.5
        constraints.append(cvx.norm(X[0:2, -1] - goal[0:2], 2) <= pos_tol)
        if self.goal_tolerances is not None:
            dir_tol = self.goal_tolerances["dir"]
            vel_tol = self.goal_tolerances["vel"]
            constraints.append(cvx.abs(X[2, -1] - goal[2]) <= dir_tol)
            constraints.append(cvx.norm(X[3:5, -1] - goal[3:5], 2) <= vel_tol)
            constraints.append(cvx.abs(X[5, -1] - goal[5]) <= vel_tol)
        # keep small thrust at end
        constraints.append(cvx.norm(U[:, -1], 2) <= 0.1)

        # ---------- Box constraints for the frame (walls) ----------
        margin = 0.8
        x_min, x_max, y_min, y_max = self._get_frame_limits(margin)

        for k in range(1, K):
            constraints += [
                X[0, k] >= x_min,
                X[0, k] <= x_max,
                X[1, k] >= y_min,
                X[1, k] <= y_max,
            ]

        # ---------- Thruster bounds (match simulator saturation) ----------
        F_min, F_max = self.sp.F_limits
        for k in range(K):
            constraints += [
                U[0, k] >= F_min,
                U[0, k] <= F_max,
                U[1, k] >= F_min,
                U[1, k] <= F_max,
            ]

        # ---------- Obstacle avoidance: planets (linearised) ----------
        for i in range(self.n_planets):
            obs_A = pp[f"obs_A_{i}"]  # shape (2, K)
            obs_b = pp[f"obs_b_{i}"]  # shape (K,)

            for k in range(K):
                a_k = obs_A[:, k]  # (2,)
                b_k = obs_b[k]  # scalar
                x_pos_k = X[0:2, k]  # (x, y)
                if s_planets is not None:
                    constraints.append(a_k @ x_pos_k + s_planets[i, k] >= b_k)
                else:
                    constraints.append(a_k @ x_pos_k >= b_k)

        # ---------- Obstacle avoidance: dynamic asteroids (linearised) ----------
        if self.n_asteroids > 0:
            for i in range(self.n_asteroids):
                ast_A = pp[f"ast_A_{i}"]  # (2, K)
                ast_b = pp[f"ast_b_{i}"]  # (K,)
                for k in range(K):
                    a_k = ast_A[:, k]
                    b_k = ast_b[k]
                    x_pos_k = X[0:2, k]
                    if s_asteroids is not None:
                        constraints.append(a_k @ x_pos_k + s_asteroids[i, k] >= b_k)
                    else:
                        constraints.append(a_k @ x_pos_k >= b_k)

        # ---------- Mild hard trust region (per step, inf-norm) ----------
        tr_x_step = self.params.tr_radius_step_x
        tr_u_step = self.params.tr_radius_step_u
        for k in range(K):
            # allow larger deviations near docking to satisfy funnel halfspaces
            tr_x_local = max(tr_x_step, 6.0) if self.docking_halfspaces and k >= self.dock_start_idx else tr_x_step
            state_dev = self.scale_mat_x @ (X[:, k] - pp["X_ref"][:, k])
            input_dev = self.scale_mat_u @ (U[:, k] - pp["U_ref"][:, k])
            constraints.append(cvx.norm(state_dev, "inf") <= tr_x_local)
            constraints.append(cvx.norm(input_dev, "inf") <= tr_u_step)

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]
        nu = self.variables["nu"]
        s_planets = self.variables.get("s_planets", None)
        s_asteroids = self.variables.get("s_asteroids", None)

        pp = self.problem_parameters

        # weights
        w_u = 0.01  # effort
        w_pos = 150.0  # position finale
        w_psi = 60.0  # orientation
        w_vel = 60.0  # vitesse finale
        w_dpsi = 25.0  # angular rate final
        w_nu = self.params.lambda_nu
        w_du = 1.0  # smooth input changes

        # ---- control effort ----
        cost_u = w_u * cvx.sum_squares(U)
        cost_du = w_du * cvx.sum_squares(U[:, 1:] - U[:, :-1])

        # ---- terminal errors ----
        goal = pp["goal_state"]
        x_final = X[:, -1]

        e_pos = x_final[0:2] - goal[0:2]
        e_psi = x_final[2] - goal[2]
        e_vel = x_final[3:5] - goal[3:5]
        e_dpsi = x_final[5] - goal[5]

        cost_pos = w_pos * cvx.sum_squares(e_pos)
        cost_psi = w_psi * cvx.sum_squares(e_psi)
        cost_vel = w_vel * cvx.sum_squares(e_vel)
        cost_dpsi = w_dpsi * cvx.sum_squares(e_dpsi)

        # ---- final time ----
        p_vec = cvx.reshape(p, (1, -1))
        cost_time = cvx.sum(self.params.weight_p @ p_vec)

        # ---- slack on dynamics ----
        cost_nu = w_nu * cvx.sum(cvx.norm1(nu, axis=0))

        # ---- obstacle slacks ----
        cost_obs = 0.0
        if s_planets is not None:
            cost_obs += self.params.lambda_obs * cvx.sum(s_planets)
        if s_asteroids is not None:
            cost_obs += self.params.lambda_obs * cvx.sum(s_asteroids)

        # ---- soft trust region to keep steps moderate ----
        w_tr_x = 0.35
        w_tr_u = 0.08
        w_tr_p = 0.05
        state_dev = self.scale_mat_x @ (X - pp["X_ref"])
        input_dev = self.scale_mat_u @ (U - pp["U_ref"])
        time_dev = cvx.multiply(self.inv_scale_p, p - pp["p_ref"])
        cost_tr = w_tr_x * cvx.sum_squares(state_dev)
        cost_tr += w_tr_u * cvx.sum_squares(input_dev)
        cost_tr += w_tr_p * cvx.sum_squares(time_dev)

        total_cost = (
            cost_u + cost_du + cost_pos + cost_psi + cost_vel + cost_dpsi + cost_time + cost_nu + cost_obs + cost_tr
        )

        return cvx.Minimize(total_cost)

    def _convexification(self):
        """
        Perform convexification step (V1):
        - compute linearization/discretization FOH around (X_bar, U_bar, p_bar)
        - update CVX parameters for dynamics, goal and obstacle half-spaces.
        """

        # 1. Discretize FOH around current reference trajectory
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        pp = self.problem_parameters

        # ----- Initial state -----
        x0 = np.array(
            [
                self.init_state.x,
                self.init_state.y,
                self.init_state.psi,
                self.init_state.vx,
                self.init_state.vy,
                self.init_state.dpsi,
            ]
        )
        pp["init_state"].value = x0

        # ----- Goal state -----
        goal = np.array(
            [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                self.goal_state.dpsi,
            ]
        )
        pp["goal_state"].value = goal

        # ----- Dynamics matrices -----
        pp["A_bar"].value = A_bar
        pp["B_plus_bar"].value = B_plus_bar
        pp["B_minus_bar"].value = B_minus_bar
        pp["F_bar"].value = F_bar
        pp["r_bar"].value = r_bar

        # ----- Reference traj (for info / possible TR) -----
        pp["X_ref"].value = self.X_bar
        pp["U_ref"].value = self.U_bar
        pp["p_ref"].value = self.p_bar
        pp["tr_radius_x"].value = self.curr_tr_x
        pp["tr_radius_u"].value = self.curr_tr_u
        pp["tr_radius_p"].value = self.curr_tr_p

        K = self.params.K

        def _linearize_circle(x_ref: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
            """
            Ensure linearization happens outside the obstacle to avoid invalid half-spaces.
            Returns (a, b) such that a^T x >= b is the supporting half-space.
            """
            d = x_ref - center
            dist = float(np.linalg.norm(d))
            if dist < radius:
                if dist < 1e-6:
                    d = np.array([radius, 0.0])
                else:
                    d = d / max(dist, 1e-6) * (radius + 0.05)
                x_ref = center + d
                dist = float(np.linalg.norm(d))

            phi_ref = dist**2 - radius**2
            grad_phi = 2.0 * d
            a = grad_phi
            b = -phi_ref + float(grad_phi @ x_ref)
            return a, b

        # ----- Linearized obstacle constraints for planets -----
        for i, pname in enumerate(self.planet_names):
            planet = self.planets[pname]
            c = np.array(planet.center, dtype=float)  # [cx, cy]
            R = float(planet.radius) + self.safe_margin  # radius + margin

            obs_A = np.zeros((2, K))
            obs_b = np.zeros(K)

            for k in range(K):
                x_ref = self.X_bar[0:2, k]
                a, b = _linearize_circle(x_ref, c, R)

                obs_A[:, k] = a
                obs_b[k] = b

            pp[f"obs_A_{i}"].value = obs_A
            pp[f"obs_b_{i}"].value = obs_b

        # ----- Linearized obstacle constraints for asteroids (dynamic) -----
        if self.n_asteroids > 0:
            # current guess for final time
            p_final = float(self.p_bar[0]) if self.p_bar.size > 0 else 0.0
            dt = p_final / max(K - 1, 1) if p_final > 0.0 else 0.0

            for i, aname in enumerate(self.asteroid_names):
                ast = self.asteroids[aname]
                start = np.array(ast.start, dtype=float)  # starting center
                vel = np.array(ast.velocity, dtype=float)  # [vx, vy]
                vel_norm = float(np.linalg.norm(vel))
                # inflate margin slightly with speed to hedge linearization error
                R = float(ast.radius) + self.safe_margin + 0.1 * vel_norm  # radius + margin

                obs_A = np.zeros((2, K))
                obs_b = np.zeros(K)

                for k in range(K):
                    t_k = k * dt
                    c_k = start + vel * t_k  # predicted asteroid center

                    x_ref = self.X_bar[0:2, k]
                    a, b = _linearize_circle(x_ref, c_k, R)

                    obs_A[:, k] = a
                    obs_b[k] = b

                pp[f"ast_A_{i}"].value = obs_A
                pp[f"ast_b_{i}"].value = obs_b

    def _check_convergence(self, step: float) -> bool:
        """
        Check convergence of SCvx.
        """

        # Converged when iterates stop moving significantly or trust region is tiny
        tr_tiny = (
            self.curr_tr_x <= self.params.min_tr_radius
            and self.curr_tr_u <= self.params.min_tr_radius
            and self.curr_tr_p <= self.params.min_tr_radius
        )
        return step < self.params.stop_crit or tr_tiny

    def _update_trust_region(self, step_x: float, step_u: float, step_p: float, success: bool):
        """
        Update trust region radius.
        """
        shrink = 1.0 / self.params.alpha
        grow = self.params.beta

        factor = 1.0
        if not success:
            factor = shrink
        else:
            near_boundary = any(
                s > 0.8 * tr for s, tr in [(step_x, self.curr_tr_x), (step_u, self.curr_tr_u), (step_p, self.curr_tr_p)]
            )
            tiny_step = all(
                s < 0.2 * tr for s, tr in [(step_x, self.curr_tr_x), (step_u, self.curr_tr_u), (step_p, self.curr_tr_p)]
            )

            if near_boundary:
                factor = grow
            elif tiny_step:
                factor = shrink

        self.curr_tr_x = float(np.clip(self.curr_tr_x * factor, self.params.min_tr_radius, self.params.max_tr_radius))
        self.curr_tr_u = float(np.clip(self.curr_tr_u * factor, self.params.min_tr_radius, self.params.max_tr_radius))
        self.curr_tr_p = float(np.clip(self.curr_tr_p * factor, self.params.min_tr_radius, self.params.max_tr_radius))

    def _setup_docking_halfspaces(self, docking_goal: DockingTarget):
        """
        Pre-compute half-space constraints to keep the approach within the docking arms cone
        and behind the docking base. This prevents grazing collisions with the docking structure.
        """
        A, B, C, A1, A2, _ = docking_goal.get_landing_constraint_points()
        margin = 0.05

        def _edge_halfspace(p0: np.ndarray, p1: np.ndarray, interior_pt: np.ndarray) -> tuple[np.ndarray, float]:
            edge = p1 - p0
            n = np.array([-edge[1], edge[0]])  # left normal
            if n @ (interior_pt - p0) < 0:
                n = -n
            b = float(n @ p0) + margin
            return n, b

        # We only keep the anchor to bias the initial guess; no extra constraints.
        self.docking_halfspaces = []
        self.docking_anchor = A
        self.dock_start_idx = self.params.K

    @staticmethod
    def _extract_seq_from_array() -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F = np.array([0, 1, 2, 3, 4])
        ddelta = np.array([0, 0, 0, 0, 0])
        cmds_list = [SatelliteCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 6)
        states = [SatelliteState(*v) for v in npstates]
        mystates = DgSampledSequence[SatelliteState](timestamps=ts, values=states)
        return mycmds, mystates


if __name__ == "_main_":
    planets = {}
    asteroids = {}

    sg = SatelliteGeometry.default()
    sp = SatelliteParameters.default()

    planner = SatellitePlanner(planets, asteroids, sg, sp)
    print("DONE.")
