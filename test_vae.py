from vae import train, VariationalAutoEncoder
from vae_cnn import ConvolutionalVariationalAutoEncoder

network_architecture = {
    "n_width": 32,
    "n_z": 20,  # dimensionality of latent space
    "n_hidden_units": 500,
    "n_layers": 3
}

vae = train(ConvolutionalVariationalAutoEncoder, network_architecture, training_epochs=300)
# vae = train(VariationalAutoEncoder, network_architecture, training_epochs=31)
