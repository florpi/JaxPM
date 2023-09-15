from pathlib import Path
from jaxpm.painting import cic_paint, cic_read

import itertools
import yaml
import argparse
from functools import partial
import jax.numpy as jnp
from jaxpm.nn import CNN, NeuralSplineFourierFilter
import sys
from absl import flags, logging
from ml_collections import config_flags

from jaxpm.pm import make_ode_fn
from jaxpm.kernels import fftk, gradient_kernel, laplace_kernel, longrange_kernel, PGD_kernel
import jax
import optax
import haiku as hk
from tqdm import tqdm
from jax import config
import jax_cosmo as jc
from jax.experimental.ode import odeint
from read_data import load_datasets
from flax.training.early_stopping import EarlyStopping
import wandb
import pickle

config.update("jax_enable_x64", True)

# EDA model for k and cnn corr when runnning with model -> Check Pk 
# Run a PM with only 3 steps
# Does batch size matter? should we average?

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
        if config.correction_model.type == 'kcorr':
            kvec = fftk((grid_data.shape[:-1]))
            pot = grid_data[...,0]
            pot_k = jnp.fft.rfftn(pot)
            kk = jnp.sqrt(sum((ki/jnp.pi)**2 for ki in kvec))
            net_output = neural_net.apply(
                params, kk, None, jnp.atleast_1d(scale_factors)
            )
            pot_k = pot_k *(1. + net_output)
            predicted_potential_grid = jnp.fft.irfftn(pot_k)
            predicted_potential = cic_read(predicted_potential_grid, pos_lr)
        elif config.correction_model.type == 'cnn':
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
            make_ode_fn(mesh_shape=(n_mesh,n_mesh,n_mesh), add_correction=config.correction_model.type, model=neural_net),
            [pos_lr[0], vel_lr[0]],
            scale_factors,
            cosmology,
            params,
            rtol=1e-5,
            atol=1e-5,
        )
        predicted_potential = jnp.stack(
            [
                get_gravitational_potential(pos_pm[i], n_mesh)[1] for i in range(len(pos_pm))
            ]
        )
        return jnp.mean((predicted_potential.squeeze() - potential_hr) ** 2)
    return loss_fn

def get_position_loss(
    neural_net,
    cosmology,
    velocity_loss=False,
):
    @jax.jit
    def loss_fn(
        params,
        grid_data,
        pos_lr,
        vel_lr,
        pos_hr,
        vel_hr,
        scale_factors,
    ):
        n_mesh = grid_data.shape[1]

        pos_pm, vel_pm = odeint(
            make_ode_fn(mesh_shape=(n_mesh,n_mesh,n_mesh), add_correction=config.correction_model.type, model=neural_net),
            [pos_lr[0], vel_lr[0]],
            scale_factors,
            cosmology,
            params,
            rtol=1e-5,
            atol=1e-5,
        )
        pos_pm %= n_mesh
        pos_hr %= n_mesh

        sim_mse = get_mse_pos(pos_pm, pos_hr, box_size=n_mesh)
        if velocity_loss:
            sim_mse += jnp.sum((vel_pm - vel_hr) ** 2, axis=-1)
        return sim_mse
    return loss_fn

def get_mse_pos(
    x,
    y,
    box_size=1.0,
):
    dx = x - y
    if box_size is not None:
        dx = dx - box_size * jnp.round(dx / box_size)
    return jnp.mean(jnp.sum(dx**2, axis=-1))


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", "config.py", "Training configuration")
    FLAGS(sys.argv)
    print('Running configuration')
    print(FLAGS.config)
    config = FLAGS.config
    if config.correction_model.type == 'cnn':
        def CorrModel(
            x,
            positions,
            scale_factors,
        ):
            kernel_size = config.correction_model.kernel_size
            cnn = CNN(
                n_channels_hidden=config.correction_model.n_channels_hidden,
                n_convolutions=config.correction_model.n_convolutions,
                n_linear=config.correction_model.n_linear,
                input_dim=config.correction_model.input_dim,
                output_dim=1,
                kernel_shape=(kernel_size, kernel_size, kernel_size),
            )
            return cnn(
                x,
                positions,
                scale_factors,
            )
    elif config.correction_model.type == 'kcorr':
        def CorrModel(
            x,
            scale_factors,
        ):
            return NeuralSplineFourierFilter(
                    n_knots=config.correction_model.n_knots, 
                    latent_size=config.correction_model.latent_size
                )(x, scale_factors)
    elif config.correction_model.type == 'cnn+kcorr':
        def CorrModel(
            x,
            positions,
            scale_factors,
        ):
            kernel_size = config.correction_model.kernel_size
            cnn = CNN(
                n_channels_hidden=config.correction_model.n_channels_hidden,
                n_convolutions=config.correction_model.n_convolutions,
                n_linear=config.correction_model.n_linear,
                input_dim=config.correction_model.input_dim,
                output_dim=1,
                kernel_shape=(kernel_size, kernel_size, kernel_size),
            )
            nsf = NeuralSplineFourierFilter(
                    n_knots=config.correction_model.n_knots, 
                    latent_size=config.correction_model.latent_size
                )(x, scale_factors)
            cnn_output = cnn(
                x,
                positions,
                scale_factors,
            )
            nsf_output = nsf(x, scale_factors)
            return cnn_output, nsf_output
    else:
        raise NotImplementedError(f'Correction model type {config.correction_model.type} not implemented')

    mesh_lr = config.data.mesh_lr
    mesh_hr = config.data.mesh_hr
    n_train_sims = config.data.n_train_sims
    n_val_sims = config.data.n_val_sims
    snapshots = config.data.snapshots
    data_dir = Path(
        f"/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/pm2nbody/data/matched_{mesh_lr}_{mesh_hr}/"
    )
    output_dir = Path(
        "/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/pm2nbody/models/"
    )
    box_size = 256.0
    loss = config.training.loss

    # *** GET DATA
    scale_factors = jnp.load(data_dir / f"scale_factors.npy")
    if snapshots is not None:
        snapshots = jnp.array(snapshots)
        scale_factors = scale_factors[snapshots]
    train_data, val_data = load_datasets(
        n_train_sims,
        n_val_sims,
        mesh_hr=mesh_hr,
        mesh_lr=mesh_lr,
        data_dir=data_dir,
        snapshots=snapshots,
    )
    print(f"Using {len(train_data)} sims for training")
    print(f"Using {len(val_data)} sims for val")

    omega_c = 0.25
    sigma8 = 0.8
    cosmology = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    # Baseline Mean squared errors


    neural_net = hk.without_apply_rng(hk.transform(CorrModel))
    rng = jax.random.PRNGKey(42)
    grid_input_init = train_data[0]["lr"].grid[0]
    pos_init = train_data[0]["lr"].positions[0] * mesh_lr
    scale_init = jnp.array([1.0])
    if config.correction_model.type == 'kcorr':
        params = neural_net.init(
            rng,
            grid_input_init,
            scale_init,
        )
    else:
        params = neural_net.init(
            rng,
            grid_input_init,
            pos_init,
            scale_init,
        )

    run = wandb.init(
        project=config.wandb.project,
        config= config.to_dict(), 
        dir = output_dir,
    )
    print(f'Run name: {run.name}')
    run_dir = output_dir / f'{run.name}'
    run_dir.mkdir(exist_ok=True, parents=True)

    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config.to_dict(), f)


    if loss == "mse_frozen_potential":
        single_loss_fn = get_frozen_potential_loss(
            neural_net=neural_net,
        )
        vmap_loss = jax.vmap(single_loss_fn, in_axes=(None, 0, 0, 0, 0,0))

        def loss_fn(params, dataset, scale_factors,):
            loss_array = vmap_loss(
                params,
                dataset["lr"].grid,
                dataset["lr"].positions * mesh_lr,
                dataset["lr"].potential,
                dataset["hr"].potential,
                scale_factors,
            )
            return jnp.mean(loss_array)

    elif loss == "mse_potential":
        single_loss_fn = get_potential_loss(
            neural_net=neural_net,
            cosmology=cosmology,
        )
        def loss_fn(params, dataset, scale_factors,):
            loss_array = single_loss_fn(
                params,
                dataset["lr"].grid,
                dataset["lr"].positions * mesh_lr,
                dataset["lr"].velocities * mesh_lr,
                dataset["hr"].potential,
                scale_factors,
            )
            return jnp.mean(loss_array)

    elif loss == "mse_positions":
        single_loss_fn = get_position_loss(
            neural_net=neural_net,
            cosmology=cosmology,
        )
        def loss_fn(params, dataset, scale_factors,):
            return single_loss_fn(
                params,
                dataset["lr"].grid,
                dataset["lr"].positions * mesh_lr,
                dataset["lr"].velocities * mesh_lr,
                dataset["hr"].positions * mesh_lr,
                dataset["hr"].velocities * mesh_lr,
                scale_factors,
            )

    batch = next(train_data.iterator)
    n_steps = 3_000  # 50_000
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.training.initial_lr, 
        peak_value=5.0e-2,
        warmup_steps=int(n_steps * 0.01),
        decay_steps=n_steps,
    )

    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=config.training.weight_decay,
    )

    opt_state = optimizer.init(params)

    val_pos_loss, val_pot_loss = [], []
    for val_batch in val_data:
        val_pos_loss.append(get_mse_pos(
            val_batch["hr"].positions * mesh_lr,
            val_batch["lr"].positions * mesh_lr,
            box_size=mesh_lr,
        ))
        val_pot_loss.append(
            jnp.mean((val_batch['lr'].potential - val_batch['hr'].potential)**2)
        )
    print('Positions MSE = ', sum(val_pos_loss) / len(val_pos_loss))
    print('Potential MSE = ', sum(val_pot_loss) / len(val_pot_loss))
    early_stop = EarlyStopping(min_delta=1e-3, patience=20)
    best_params = None
    pbar = tqdm(range(n_steps))
    for step in pbar:
        def train_loss_fn(params,):
            batch = next(train_data.iterator)
            return loss_fn(
                params=params,
                dataset=batch,
                scale_factors=scale_factors,
            )

        train_loss, grads = jax.value_and_grad(
            train_loss_fn, 
        )(params,)
        updates, opt_state = optimizer.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)
        pbar.set_postfix(
            {
                "Step": step,
                "Loss": train_loss,
            }
        )
        if step % 5 == 0:
            val_loss = 0.0
            for val_batch in val_data:
                val_loss += loss_fn(params, val_batch, scale_factors,)
            val_loss /= len(val_data)
            has_improved, early_stop = early_stop.update(val_loss)
            if has_improved:
                best_params = params

            wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=step,)

            pbar.set_postfix(val_loss=val_loss)

            should_stop, early_stop = early_stop.update(val_loss)
            if early_stop.should_stop:
                break

    best_loss = early_stop.best_metric
    with open(run_dir / f"{best_loss:.3f}_weights.pkl", "wb") as f:
        state_dict = hk.data_structures.to_immutable_dict(best_params)
        pickle.dump(state_dict, f)
