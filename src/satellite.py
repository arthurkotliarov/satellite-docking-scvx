import sympy as spy

from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters
from typing import Any, Callable, Tuple, cast


class SatelliteDyn:
    sg: SatelliteGeometry
    sp: SatelliteParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SatelliteGeometry, sp: SatelliteParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust_l thrust_r", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        extract the state from self.x the following way:
        0x 1y 2psi 3vx 4vy 5dpsi
        """

        f = spy.zeros(self.n_x, 1)
        x = self.x
        u = self.u
        p = self.p

        psi = cast(spy.Expr, x[2])
        vx = cast(spy.Expr, x[3])
        vy = cast(spy.Expr, x[4])
        dpsi = cast(spy.Expr, x[5])

        Fl = cast(Any, u[0])
        Fr = cast(Any, u[1])
        m = cast(Any, self.sp.m_v)
        I = cast(Any, self.sg.Iz)
        l = cast(Any, self.sg.l_m)
        t_f = cast(Any, p[0])

        dx = vx
        dy = vy
        dpsi_dt = dpsi
        dvx = spy.cos(psi) * (Fr + Fl) / m
        dvy = spy.sin(psi) * (Fr + Fl) / m
        ddpsi = l * (Fr - Fl) / I

        f[0] = t_f * dx
        f[1] = t_f * dy
        f[2] = t_f * dpsi_dt
        f[3] = t_f * dvx
        f[4] = t_f * dvy
        f[5] = t_f * ddpsi

        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func
