import jax
import jax.numpy as jnp

import jax_cosmo as jc

from jaxpm.kernels import (
    fftk,
    gradient_kernel,
    laplace_kernel,
    longrange_kernel,
    PGD_kernel,
)
from jaxpm.painting import cic_paint, cic_read
from jaxpm.growth import growth_factor, growth_rate, dGfa


def get_delta(
    positions,
    mesh_shape,
):
    cell_volume = 1.0 / jnp.prod(jnp.array(mesh_shape))
    normalization = cell_volume * len(positions)
    delta = cic_paint(jnp.zeros(mesh_shape), positions)
    delta /= normalization
    return delta


def potential_kgrid_to_force_at_pos(
    delta_k,
    positions,
    kvec,
    r_split=0,
    return_potential=False,
):
    pot_k = delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=r_split)
    force = jnp.stack(
        [
            cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k), positions)
            for i in range(3)
        ],
        axis=-1,
    )
    if return_potential:
        pot = jnp.fft.irfftn(pot_k)
        return force, pot
    return force


def get_corrected_potential_fn(model, params, grid_data, a):
    def get_corrected_potential(positions):
        return model.apply(params, grid_data, positions, a).squeeze()

    return get_corrected_potential


def pm_forces(
    positions,
    mesh_shape=None,
    delta=None,
    r_split=0,
    add_correction=None,
    a=None,
    model=None,
    params=None,
):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if mesh_shape is None:
        mesh_shape = delta.shape
    if delta is None:
        delta = get_delta(positions, mesh_shape)
        delta_k = jnp.fft.rfftn(delta)
    else:
        delta_k = jnp.fft.rfftn(delta)
    kvec = fftk(mesh_shape)
    if add_correction is None:
        return potential_kgrid_to_force_at_pos(
            delta_k=delta_k,
            kvec=kvec,
            positions=positions,
            r_split=r_split,
        )
    elif add_correction == "cnn":
        pm_force, pm_pot = potential_kgrid_to_force_at_pos(
            delta_k=delta_k,
            kvec=kvec,
            positions=positions,
            r_split=r_split,
            return_potential=True,
        )
        grid_data = jnp.stack([pm_pot, delta], axis=-1)
        get_corrected_potential = get_corrected_potential_fn(model, params, grid_data, a)
        corrected_potential_grad = jax.vmap(
            jax.grad(get_corrected_potential),
            in_axes=(0),
        )(positions)
        pm_force += corrected_potential_grad
        return pm_force
    elif add_correction == "kcorr":
        kk = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
        delta_k = delta_k * (1.0 + model.apply(params, kk, jnp.atleast_1d(a)))
        return potential_kgrid_to_force_at_pos(
            delta_k=delta_k,
            positions=positions,
            kvec=kvec,
            r_split=r_split,
        )

    else:
        raise NotImplementedError(f"add_correction={add_correction} not implemented")


def lpt(cosmo, initial_conditions, positions, a):
    """
    Computes first order LPT displacement
    """
    initial_force = pm_forces(positions, delta=initial_conditions)
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dx
    f = a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dGfa(cosmo, a) * initial_force
    return dx, p, f


def linear_field(mesh_shape, box_size, pk, seed):
    """
    Generate initial conditions.
    """
    kvec = fftk(mesh_shape)
    kmesh = (
        sum((kk / box_size[i] * mesh_shape[i]) ** 2 for i, kk in enumerate(kvec)) ** 0.5
    )
    pkmesh = (
        pk(kmesh)
        * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2])
        / (box_size[0] * box_size[1] * box_size[2])
    )

    field = jax.random.normal(seed, mesh_shape)
    field = jnp.fft.rfftn(field) * pkmesh**0.5
    field = jnp.fft.irfftn(field)
    return field


def make_ode_fn(
    mesh_shape,
    add_correction=None,
    model=None,
):
    def nbody_ode(state, a, cosmo, params=None,):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state
        forces = (
            1.5
            * cosmo.Omega_m
            * pm_forces(
                positions=pos,
                mesh_shape=mesh_shape,
                add_correction=add_correction,
                model=model,
                params=params,
                a=a,
            )
        )
        # Computes the update of position (drift)
        dpos = 1.0 / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1.0 / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces
        return dpos, dvel

    return nbody_ode


def pgd_correction(pos, params):
    """
    improve the short-range interactions of PM-Nbody simulations with potential gradient descent method, based on https://arxiv.org/abs/1804.00671
    args:
      pos: particle positions [npart, 3]
      params: [alpha, kl, ks] pgd parameters
    """
    kvec = fftk(mesh_shape)

    delta = cic_paint(jnp.zeros(mesh_shape), pos)
    alpha, kl, ks = params
    delta_k = jnp.fft.rfftn(delta)
    PGD_range = PGD_kernel(kvec, kl, ks)

    pot_k_pgd = (delta_k * laplace_kernel(kvec)) * PGD_range

    forces_pgd = jnp.stack(
        [
            cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k_pgd), pos)
            for i in range(3)
        ],
        axis=-1,
    )

    dpos_pgd = forces_pgd * alpha

    return dpos_pgd
