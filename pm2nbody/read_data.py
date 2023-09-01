from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import jax.numpy as jnp


def downsample_to_mesh(
    array: jnp.array, n_mesh: int, downsampling_factor: int
) -> jnp.array:
    """Downsample an array to the number of particles in a lower resolution mesh, such
    that low and high resolution particles match

    """
    first_dim_array = len(array)
    if len(array.shape) == 3:
        last_dim_array = array.shape[-1]
        first_reshape_to = (first_dim_array, n_mesh, n_mesh, n_mesh, last_dim_array)
        last_reshape_to = (first_dim_array, -1, last_dim_array)
    elif len(array.shape) == 2:
        first_reshape_to = (
            first_dim_array,
            n_mesh,
            n_mesh,
            n_mesh,
        )
        last_reshape_to = (first_dim_array, -1)
    else:
        raise ValueError("Array must be 2 or 3 dimensional")
    return array.reshape(first_reshape_to)[
        :,
        ::downsampling_factor,
        ::downsampling_factor,
        ::downsampling_factor,
    ].reshape(last_reshape_to)


def get_data(
    data_dir: Path,
    n_mesh: int,
    downsampling_factor: Optional[int] = None,
    get_grids: Optional[bool] = False,
    snapshots: Optional[List[int]] = None,
    box_size: Optional[float] = 256.0,
    normalize_pos_to_box: Optional[bool] = True,
    idx: Optional[int] = 0,
):
    pos = jnp.load(data_dir / f"pos_m{n_mesh}_s{idx}.npy")
    if snapshots is None:
        snapshots = jnp.arange(len(pos))
    pos = pos[snapshots, :, :]
    if normalize_pos_to_box:
        pos /= box_size
    vel = jnp.load(data_dir / f"vel_m{n_mesh}_s{idx}.npy")[snapshots]
    gravitational_potential = jnp.load(data_dir / f"pot_m{n_mesh}_s{idx}.npy")[
        snapshots
    ]
    if downsampling_factor is not None:
        pos = downsample_to_mesh(
            array=pos,
            n_mesh=n_mesh,
            downsampling_factor=downsampling_factor,
        )
        vel = downsample_to_mesh(
            array=vel,
            n_mesh=n_mesh,
            downsampling_factor=downsampling_factor,
        )
        gravitational_potential = downsample_to_mesh(
            array=gravitational_potential,
            n_mesh=n_mesh,
            downsampling_factor=downsampling_factor,
        )
    if not get_grids:
        return pos, vel, gravitational_potential
    potential_grid = jnp.load(data_dir / f"pot_grid_m{n_mesh}_s{idx}.npy")[snapshots]
    density_grid = jnp.load(data_dir / f"pot_grid_m{n_mesh}_s{idx}.npy")[snapshots]
    return pos, vel, gravitational_potential, potential_grid, density_grid


@dataclass
class ResolutionData:
    positions: jnp.array
    velocities: jnp.array
    potential: jnp.array
    potential_grid: Optional[jnp.array] = None
    density_grid: Optional[jnp.array] = None
    grid: Optional[jnp.array] = None

    def __post_init__(
        self,
    ):
        if self.potential_grid is not None and self.density_grid is not None:
            self.grid = jnp.stack([self.potential_grid, self.density_grid], axis=-1)


class PMDataset:
    def __init__(self, high_res_data, low_res_data):
        self.hr = high_res_data
        self.lr = low_res_data
        self.iterator = iter(self)

    def __len__(
        self,
    ):
        return len(self.hr)

    def __getitem__(self, idx):
        high_res_data = self.hr[idx]
        low_res_data = self.lr[idx]
        return {"hr": high_res_data, "lr": low_res_data}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def load_dataset_for_sim_idx_list(idx_list, mesh_hr, mesh_lr, data_dir):
    grid_factor = mesh_hr / mesh_lr
    particle_factor = mesh_hr / mesh_lr
    up_resolution_factor = grid_factor * particle_factor

    low_res_data, high_res_data = [], []
    for idx in idx_list:
        pos_hr, vel_hr, grav_pot_hr = get_data(
            data_dir=data_dir,
            n_mesh=mesh_hr,
            downsampling_factor=mesh_hr // mesh_lr,
            idx=idx,
        )
        pos_lr, vel_lr, grav_pot_lr, grav_pot_grid_lr, dens_grid_lr = get_data(
            data_dir=data_dir,
            n_mesh=mesh_lr,
            get_grids=True,
            idx=idx,
        )
        grav_pot_grid_lr *= up_resolution_factor
        dens_grid_lr *= up_resolution_factor
        grav_pot_lr *= up_resolution_factor
        high_res_data.append(ResolutionData(pos_hr, vel_hr, grav_pot_hr, None, None))
        low_res_data.append(
            ResolutionData(pos_lr, vel_lr, grav_pot_lr, grav_pot_grid_lr, dens_grid_lr)
        )
    return low_res_data, high_res_data


def load_datasets(n_train_sims, n_val_sims, mesh_hr, mesh_lr, data_dir):
    train_idx_list = list(range(n_train_sims))
    val_idx_list = list(range(n_train_sims, n_train_sims + n_val_sims))
    train_low_res_data, train_high_res_data = load_dataset_for_sim_idx_list(
        train_idx_list, mesh_hr, mesh_lr, data_dir
    )
    val_low_res_data, val_high_res_data = load_dataset_for_sim_idx_list(
        val_idx_list, mesh_hr, mesh_lr, data_dir
    )
    return PMDataset(train_high_res_data, train_low_res_data), PMDataset(
        val_high_res_data, val_low_res_data
    )
