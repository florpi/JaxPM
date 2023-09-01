from pathlib import Path
import argparse
from functools import partial
import jax.numpy as jnp
from jaxpm.nn import CNN

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
    parser = argparse.ArgumentParser()
    parser.parse_args()
    parser.add_argument(
        "--n_channels_hidden",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--n_convolutions",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--n_linear",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--mesh_lr",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--mesh_hr",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--n_train_sims",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--n_val_sims",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse_potential",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1.0e-5,
    )
    args = parser.parse_args()

    def ConvNet(
        x,
        positions,
        scale_factors,
    ):
        cnn = CNN(
            n_channels_hidden=args.n_channels_hidden,
            n_convolutions=args.n_convolutions,
            n_linear=args.n_linear,
            input_dim=args.input_dim,
            output_dim=1,
            kernel_shape=(args.kernel_size, args.kernel_size, args.kernel_size),
        )
        return cnn(
            x,
            positions,
            scale_factors,
        )

    mesh_lr = args.mesh_lr
    mesh_hr = args.mesh_hr
    n_train_sims = args.n_train_sims
    n_val_sims = args.n_val_sims
    data_dir = Path(
        f"/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/pm2nbody/data/matched_{mesh_lr}_{mesh_hr}/"
    )
    output_dir = Path(
        "/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/pm2nbody/models/"
    )
    box_size = 256.0
    loss = args.loss

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
        project="pm2nbody",
        config= vars(args),
        dir = output_dir,
    )
    print(f'Run name: {run.name}')
    run_dir = output_dir / f'{run.name}'
    run_dir.mkdir(exist_ok=True, parents=True)

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
        weight_decay=args.weight_decay,
    )

    losses = []
    opt_state = optimizer.init(params)

    early_stop = EarlyStopping(min_delta=1e-3, patience=4)
    pbar = tqdm(range(n_steps))
    for step in pbar:

        def train_loss_fn(params, model_state):
            batch = next(train_data.iterator)
            return loss_fn(
                params,
                model_state,
                batch,
                scale_factors,
            )

        (train_loss, model_state), grads = jax.value_and_grad(
            train_loss_fn, has_aux=True
        )(params, model_state)
        updates, opt_state = optimizer.update(grads, opt_state)

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
                vl, _ = loss_fn(params, model_state, val_batch)
                val_loss += vl
            val_loss /= len(val_data)
            _, early_stop = early_stop.update(val_loss)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=step,)

            pbar.set_postfix(val_loss=val_loss)
            if early_stop.should_stop:
                print("Early stopping")
                break

    params = early_stop.best_params
    with open(run_dir / f"{loss}_{train_loss:.3f}_weights.pkl", "wb") as f:
        state_dict = hk.data_structures.to_immutable_dict(params)
        pickle.dump(state_dict, f)
