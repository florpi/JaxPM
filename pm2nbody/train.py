from pathlib import Path
import itertools
import yaml
import argparse
from functools import partial
import jax.numpy as jnp
from jaxpm.nn import CNN, NeuralSplineFourierFilter
import sys
from absl import flags, logging
from ml_collections import config_flags

# from jaxpm.pm import make_hamiltonian_ode_fn
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


def get_frozen_potential_loss(
    neural_net,
):
    @jax.jit
    def loss_fn(
        params,
        model_state,
        grid_data,
        pos_lr,
        scale_factors,
        potential_lr,
        potential_hr,
    ):
        corrected_potential, model_state = neural_net.apply(
            params,
            model_state,
            grid_data,
            pos_lr,
            scale_factors,
        )
        predicted_potential = corrected_potential.squeeze() + potential_lr
        return jnp.mean((predicted_potential - potential_hr) ** 2), model_state

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

    def ConvNet(
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

    mesh_lr = config.data.mesh_lr
    mesh_hr = config.data.mesh_hr
    n_train_sims = config.data.n_train_sims
    n_val_sims = config.data.n_val_sims
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
    train_data, val_data = load_datasets(
        n_train_sims,
        n_val_sims,
        mesh_hr=mesh_hr,
        mesh_lr=mesh_lr,
        data_dir=data_dir,
    )
    print(f"Using {len(train_data)} sims for training")
    print(f"Using {len(val_data)} sims for val")

    omega_c = 0.25
    sigma8 = 0.8
    cosmology = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    # Baseline Mean squared errors
    print(
        "Positions MSE = ",
        get_mse_pos(
            train_data[0]["hr"].positions,
            train_data[0]["lr"].positions,
            box_size=1.0,
        ),
    )
    print(
        "Potential MSE = ",
        jnp.mean((train_data[0]["hr"].potential - train_data[0]["lr"].potential) ** 2),
    )

    neural_net = hk.without_apply_rng(hk.transform_with_state(ConvNet))
    rng = jax.random.PRNGKey(42)
    grid_input_init = train_data[0]["lr"].grid[0]
    pos_init = train_data[0]["lr"].positions[0] * mesh_lr
    params, model_state = neural_net.init(
        rng,
        grid_input_init,
        pos_init,
        jnp.array([1.0]),
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


    if loss == "mse_potential":
        single_loss_fn = get_frozen_potential_loss(
            neural_net=neural_net,
        )
        vmap_loss = jax.vmap(single_loss_fn, in_axes=(None, None, 0, 0, 0, None, None))

        def loss_fn(params, model_state, dataset, scale_factors):
            loss_array, model_state = vmap_loss(
                params,
                model_state,
                dataset["lr"].grid,
                dataset["lr"].positions * mesh_lr,
                scale_factors,
                dataset["lr"].potential,
                dataset["hr"].potential,
            )
            return jnp.mean(loss_array), model_state

    learning_rate = 5.0e-2
    n_steps = 3_000  # 50_000
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0001,
        peak_value=5.0e-2,
        warmup_steps=int(n_steps * 0.2),
        decay_steps=n_steps,
    )

    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=config.training.weight_decay,
    )

    opt_state = optimizer.init(params)

    early_stop = EarlyStopping(min_delta=1e-3, patience=4)
    best_params = None
    pbar = tqdm(range(n_steps))
    for step in pbar:
        def train_loss_fn(params, model_state):
            batch = next(train_data.iterator)
            return loss_fn(
                params=params,
                model_state=model_state,
                dataset=batch,
                scale_factors=scale_factors,
            )

        (train_loss, model_state), grads = jax.value_and_grad(
            train_loss_fn, has_aux=True
        )(params, model_state)
        updates, opt_state = optimizer.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)
        pbar.set_postfix(
            {
                "Step": step,
                "Loss": train_loss,
            }
        )
        if step % 10 == 0:
            val_loss = 0.0
            for val_batch in val_data:
                vl, _ = loss_fn(params, model_state, val_batch, scale_factors)
                val_loss += vl
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
    with open(run_dir / f"{best_loss}_{train_loss:.3f}_weights.pkl", "wb") as f:
        state_dict = hk.data_structures.to_immutable_dict(best_params)
        pickle.dump(state_dict, f)
