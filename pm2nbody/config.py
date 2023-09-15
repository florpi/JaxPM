import dataclasses
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "pm2nbody"
    wandb.log_train = True
    wandb.name = None

    config.data = data = ml_collections.ConfigDict()
    data.mesh_lr = 32
    data.mesh_hr =  64
    data.n_train_sims = 20
    data.n_val_sims = 2
    data.snapshots = None

    config.training = training = ml_collections.ConfigDict()
    training.loss = 'mse_positions'
    training.weight_decay = 1.e-3
    training.initial_lr = 0.0001 #0.0001,

    config.correction_model = correction_model = ml_collections.ConfigDict()
    #correction_model.type = 'cnn'
    correction_model.n_channels_hidden = 16
    correction_model.n_convolutions = 3
    correction_model.n_linear = 2
    correction_model.input_dim = 2
    correction_model.kernel_size = 3
    correction_model.type = 'kcorr'
    correction_model.n_knots = 16
    correction_model.latent_size = 32
    return config
