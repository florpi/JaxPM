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
    data.mesh_lr = 64
    data.mesh_hr = 128
    data.n_train_sims = 2
    data.n_val_sims = 2

    config.training = training = ml_collections.ConfigDict()
    training.loss = 'mse_potential'
    training.weight_decay = 1.e-5

    config.correction_model = correction_model = ml_collections.ConfigDict()
    correction_model.type = 'cnn'
    correction_model.n_channels_hidden = 16
    correction_model.n_convolutions = 3
    correction_model.n_linear = 2
    correction_model.input_dim = 2
    correction_model.kernel_size = 3
    return config