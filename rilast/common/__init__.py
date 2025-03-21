# This file makes the 'common' folder a package.
from .vae_training_utils import VecVAE, VecVAESimple, DeepVecVAE
from .vae_training_utils import train_step, test_step, save_model, load_model


VAE = {
    "VAE2": VecVAE,
    "VAE1": VecVAESimple,
    "VAE3": DeepVecVAE
}