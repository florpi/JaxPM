import dataclasses
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "pm2nbody"
    wandb.log_train = False 
    wandb.name = None

    config.data = data = ml_collections.ConfigDict()
    data.mesh_lr = 64
    data.mesh_hr = 128
    data.n_train_sims = 4
    data.n_val_sims = 3
    data.box_size = 256.
    data.n_snapshots = 25
    data.snapshots = None

    config.training = training = ml_collections.ConfigDict()
    training.loss = "mse_positions"
    training.weight_snapshots = False 
    training.weight_decay = 1.e-9 
    training.initial_lr = 0.01  # 0.0001,
    training.batch_size = 2
    training.n_steps = 500 
    training.checkpoint_every = 20
    training.patience = 10

    config.correction_model = correction_model = ml_collections.ConfigDict()
    correction_model.type = 'kcorr'
    correction_model.n_knots = 16
    correction_model.latent_size = 32
    return config
