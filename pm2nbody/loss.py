import jax
import jax.numpy as jnp
from jaxpm.painting import cic_paint, cic_read
from jaxpm.kernels import (
    fftk,
    gradient_kernel,
    laplace_kernel,
    longrange_kernel,
    PGD_kernel,
)
from jax.experimental.ode import odeint
import jax_cosmo.background as bkgrd 
from jaxpm.pm import make_ode_fn


def get_gravitational_potential(
    pos,
    n_mesh,
):
    mesh_shape = (n_mesh, n_mesh, n_mesh)
    kvec = fftk(mesh_shape)
    delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), pos))
    pot_k = -delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)
    pot_grid = 0.5 * jnp.fft.irfftn(pot_k)
    return pot_grid, cic_read(pot_grid, pos)


def get_frozen_potential_loss(
    neural_net,
):
    @jax.jit
    def loss_fn(
        params,
        grid_data,
        pos_lr,
        potential_lr,
        potential_hr,
        scale_factors,
    ):
        if config.correction_model.type == "kcorr":
            kvec = fftk((grid_data.shape[:-1]))
            pot = grid_data[..., 0]
            pot_k = jnp.fft.rfftn(pot)
            kk = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
            net_output = neural_net.apply(
                params, kk, None, jnp.atleast_1d(scale_factors)
            )
            pot_k = pot_k * (1.0 + net_output)
            predicted_potential_grid = jnp.fft.irfftn(pot_k)
            predicted_potential = cic_read(predicted_potential_grid, pos_lr)
        elif config.correction_model.type == "cnn":
            corrected_potential = neural_net.apply(
                params,
                grid_data,
                pos_lr,
                scale_factors,
            )
            predicted_potential = corrected_potential.squeeze() + potential_lr
        return jnp.mean((predicted_potential - potential_hr) ** 2)

    return loss_fn


def get_potential_loss(
    neural_net,
    cosmology,
):
    @jax.jit
    def loss_fn(
        params,
        grid_data,
        pos_lr,
        vel_lr,
        potential_hr,
        scale_factors,
    ):
        n_mesh = grid_data.shape[1]

        pos_pm, vel_pm = odeint(
            make_ode_fn(
                mesh_shape=(n_mesh, n_mesh, n_mesh),
                add_correction=config.correction_model.type,
                model=neural_net,
            ),
            [pos_lr[0], vel_lr[0]],
            scale_factors,
            cosmology,
            params,
            rtol=1e-5,
            atol=1e-5,
        )
        predicted_potential = jnp.stack(
            [
                get_gravitational_potential(pos_pm[i], n_mesh)[1]
                for i in range(len(pos_pm))
            ]
        )
        return jnp.mean((predicted_potential.squeeze() - potential_hr) ** 2)

    return loss_fn


def get_position_loss(
    neural_net,
    cosmology,
    n_mesh: int,
    velocity_loss=False,
    correction_type = None,
    weight_snapshots = False,
):
    @jax.jit
    def loss_fn(
        params,
        pos_lr,
        vel_lr,
        pos_hr,
        vel_hr,
        scale_factors,
    ):

        pos_pm, vel_pm = odeint(
            make_ode_fn(
                mesh_shape=(n_mesh, n_mesh, n_mesh),
                add_correction=correction_type,
                model=neural_net,
            ),
            [pos_lr[0], vel_lr[0]],
            scale_factors,
            cosmology,
            params,
            rtol=1e-5,
            atol=1e-5,
        )
        pos_pm %= n_mesh
        pos_hr %= n_mesh

        if weight_snapshots:
            #snapshot_weights = 1./bkgrd.growth_factor(cosmology, scale_factors)[:,None] 
            snapshot_weights = (1./scale_factors**1.5)[:,None]
        else:
            snapshot_weights = None
        sim_mse = get_mse_pos(pos_pm, pos_hr, box_size=n_mesh, snapshot_weights=snapshot_weights,)
        if velocity_loss:
            sim_mse += jnp.sum((vel_pm - vel_hr) ** 2, axis=-1)
        return sim_mse

    return loss_fn


def get_mse_pos(
    x,
    y,
    box_size=1.0,
    snapshot_weights=None,
):
    dx = x - y
    if box_size is not None:
        dx = dx - box_size * jnp.round(dx / box_size)
    if snapshot_weights is not None:
        return jnp.mean(snapshot_weights * jnp.sum(dx**2, axis=-1))
    return jnp.mean(jnp.sum(dx**2, axis=-1))
