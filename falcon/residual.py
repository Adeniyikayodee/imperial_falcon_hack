"""Learned aerodynamic residuals. Both map (params, v, q, thrust_ratio) to a world-frame force.

ph:  F = -D(z) (v - w_hat),  D = R sat(L L^T + diag(softplus(d)) + eps I) R^T
     sat(A) = d_max A (d_max I + A)^-1 maps eigenvalues l to d_max l / (d_max + l), so for any
     params 0 < D < d_max I. With H = 1/2 m |v - w_hat|^2 this adds a bounded dissipation matrix
     to the port-Hamiltonian dynamics: the model never injects energy relative to its air-mass
     estimate, and the dissipation matrix itself stays below the controller's m Kv velocity gain.
     How D varies with velocity is not bounded, so the full Jacobian of F can still exceed it.
     Each component of w_hat stays inside (-w_max, w_max).
mlp: F = R mlp(R^T v, thrust_ratio) + b.  Unconstrained baseline.
"""
import jax
import jax.numpy as jnp

from falcon.dynamics import quat_to_rot

EPS = 1e-4
W_MAX = 12.0  # m/s, bound on the wind estimate
D_MAX = 0.1  # N s/m, below the controller's m Kv = 0.216
TRIL = jnp.tril_indices(3)


def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    layers = [(jax.random.normal(k, (a, b)) / jnp.sqrt(a), jnp.zeros(b)) for k, a, b in zip(keys, sizes, sizes[1:])]
    layers[-1] = (jnp.zeros_like(layers[-1][0]), layers[-1][1])  # start at zero output
    return layers


def mlp(layers, z):
    for W, b in layers[:-1]:
        z = jnp.tanh(z @ W + b)
    W, b = layers[-1]
    return z @ W + b


def wind_hat(params):
    return W_MAX * jnp.tanh(params["w"])  # unitless param: preconditions the tiny dF/dw = D


def ph_init(key, hidden=32, drag0=0.01):
    """Initial A = drag0 * I (D slightly below, after saturation), split between L L^T and the diagonal. L starts away from zero,
    since d(L L^T)/dL vanishes at L = 0 and the net would never learn."""
    net = init_mlp(key, [5, hidden, hidden, 6])
    W, b = net[-1]
    net[-1] = (W, b.at[jnp.array([0, 2, 5])].set(jnp.sqrt(drag0 / 2)))
    d0 = jnp.log(jnp.expm1(drag0 / 2))  # softplus(d0) = drag0 / 2
    return {"net": net, "d": jnp.full(3, d0), "w": jnp.zeros(3)}


def ph_damping(params, v, q, thrust_ratio):
    R = quat_to_rot(q)
    v_rel_b = R.T @ (v - wind_hat(params))
    speed = jnp.sqrt(v_rel_b @ v_rel_b + 1e-6)  # smooth at zero: plain norm has a NaN gradient there
    z = jnp.concatenate([v_rel_b, jnp.array([thrust_ratio, speed])])
    L = jnp.zeros((3, 3)).at[TRIL].set(mlp(params["net"], z))
    A = L @ L.T + jnp.diag(jax.nn.softplus(params["d"]) + EPS)
    D_b = D_MAX * jnp.linalg.solve(D_MAX * jnp.eye(3) + A, A)
    return R @ (0.5 * (D_b + D_b.T)) @ R.T


def ph_apply(params, v, q, thrust_ratio):
    return -ph_damping(params, v, q, thrust_ratio) @ (v - wind_hat(params))


def mlp_init(key, hidden=32):
    return {"net": init_mlp(key, [4, hidden, hidden, 3]), "b": jnp.zeros(3)}


def mlp_apply(params, v, q, thrust_ratio):
    R = quat_to_rot(q)
    z = jnp.concatenate([R.T @ v, jnp.array([thrust_ratio])])
    return R @ mlp(params["net"], z) + params["b"]


def zero_apply(params, v, q, thrust_ratio):
    return jnp.zeros(3)


MODELS = {
    "none": (lambda key: {}, zero_apply),
    "mlp": (mlp_init, mlp_apply),
    "ph": (ph_init, ph_apply),
}
