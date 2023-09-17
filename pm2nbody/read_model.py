from pathlib import Path
import yaml
import pickle
import haiku as hk
from jaxpm.nn import CNN, NeuralSplineFourierFilter


def get_cnn_neural_net(config):
    def ConvNet(
        x,
        positions,
        scale_factors,
    ):
        cnn = CNN(
            channels_hidden_dim=config["channels_hidden_dim"],
            n_convolutions=config["n_convolutions"],
            n_fully_connected=config["n_fully_connected"],
            n_globals_embedding=config["n_globals_embedding"],
            globals_embedding_dim=config["globals_embedding_dim"],
            embed_globals=config["embed_globals"],
            input_dim=config["input_dim"],
            pad_periodic=config["pad_periodic"],
            output_dim=1,
            kernel_size=config["kernel_size"],
        )
        return cnn(
            x,
            positions,
            scale_factors,
        )

    neural_net = hk.without_apply_rng(hk.transform(ConvNet))
    return neural_net


def get_kcorr_neural_net(config):
    def FourierModel(
        x,
        scale_factors,
    ):
        return NeuralSplineFourierFilter(
            n_knots=config["n_knots"],
            latent_size=config["latent_size"],
        )(x, scale_factors)

    neural_net = hk.without_apply_rng(hk.transform(FourierModel))
    return neural_net


def read_model(
    path_to_model: Path,
):
    pkl_file = list(path_to_model.glob("best*.pkl"))[0]
    with open(pkl_file, "rb") as f:
        params = pickle.load(f)

    # open yaml file
    with open(path_to_model / "config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["correction_model"]
    if config["type"] == "cnn":
        neural_net = get_cnn_neural_net(config)
    elif config["type"] == "kcorr":
        neural_net = get_kcorr_neural_net(config)
    else:
        raise ValueError(f'Unknown correction model type: {config["type"]}')
    return neural_net, params
