import dataclasses
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "pm2nbody_small"
    wandb.log_train = False
    wandb.name = None

    config.data = data = ml_collections.ConfigDict()
    data.mesh_lr = 32
    data.mesh_hr = 64
    data.n_train_sims = 4
    data.n_val_sims = 3
    data.box_size = 256.0
    data.n_snapshots = 25
    data.n_particles = 32
    data.snapshots = None

    config.training = training = ml_collections.ConfigDict()
    training.loss = "mse_positions"
    training.log_pos = False
    training.lambda_pos = 1.0
    training.lambda_velocity = None
    training.lambda_density = None
    training.lambda_cross_corr = None
    training.weight_snapshots = False
    training.sample_snapshots = False
    training.weight_decay = 1.0e-5
    training.batch_size = 2
    training.n_steps = 800
    training.checkpoint_every = 20
    training.patience = 10
    training.schedule = schedule = ml_collections.ConfigDict()
    schedule.type = "plateau"
    schedule.initial_lr = 0.001
    schedule.factor = 0.5
    schedule.patience = 10
    schedule.min_lr = 1.0e-6

    config.correction_model = correction_model = ml_collections.ConfigDict()
    correction_model.type = "cnn"
    correction_model.channels_hidden_dim = 16
    correction_model.n_convolutions = 3
    correction_model.n_fully_connected = 3
    correction_model.input_dim = 2
    correction_model.kernel_size = 3
    correction_model.pad_periodic = True
    correction_model.embed_globals = True
    correction_model.global_conditioning = None
    correction_model.n_globals_embedding = 3
    correction_model.globals_embedding_dim = 16
    correction_model.n_knots = 16
    correction_model.latent_size = 32
    return config
