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
        kernel_size = config['kernel_size']
        cnn = CNN(
            n_channels_hidden=config['n_channels_hidden'],
            n_convolutions=config['n_convolutions'],
            n_linear=config['n_linear'],
            input_dim=config['input_dim'],
            output_dim=1,
            kernel_shape=(kernel_size, kernel_size, kernel_size),
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
                n_knots=config['n_knots'], 
                latent_size=config['latent_size'],
            )(x, scale_factors)
    neural_net = hk.without_apply_rng(hk.transform(FourierModel))
    return neural_net

def read_model(
    path_to_model: Path,
):
    pkl_file = list(path_to_model.glob('*.pkl'))[0]
    with open(pkl_file, 'rb') as f:
        params = pickle.load(f)

    # open yaml file
    with open(path_to_model / 'config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['correction_model']
    if config['type'] == 'cnn':
        neural_net = get_cnn_neural_net(config)
    elif config['type'] == 'kcorr':
        neural_net = get_kcorr_neural_net(config)
    else:
        raise ValueError(f'Unknown correction model type: {config["type"]}')
    return neural_net, params

    