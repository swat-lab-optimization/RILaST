import sys
import argparse
import torch
import torch.utils.data
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import importlib
import numpy as np
from rilast.generators.kappa_generator import KappaRoadGenerator
from rilast.generators.abstract_generator import AbstractGenerator
from rilast.validators.abstract_validator import AbstractValidator


def load_model(epoch, model, path=".//"):

    # creating the file name indexed by the epoch value
    # filename = path + '\\neural_network_{}.pt'.format(epoch)
    filename = os.path.join(path, "neural_network_{}.pt".format(epoch))

    # loading the parameters of the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(filename, map_location=device, weights_only=True))

    return model


def sample_dataset(
    num_samples: int,
    model: torch.nn.Module,
    dataset: np.ndarray,
    gen: AbstractGenerator,
    save_path: str,
    validator: AbstractValidator = None,
) -> None:
    """
    Samples a dataset using a trained model, visualizes results, and computes reconstruction errors.

    Args:
        num_samples (int): Number of samples to generate.
        model (torch.nn.Module): Trained model with an encoder-decoder architecture.
        device (torch.device): Device to run the model on (CPU/GPU).
        dataset (np.ndarray): Input dataset.
        gen (AbstractGenerator): Generator object for visualizations and transformations.
        save_path (str): Directory path to save visualization results.
        validator (Optional[object]): Validator object for checking sample validity.
    """

    model.eval()  # Set the model to evaluation mode

    print(f"Saving results to: {save_path}")

    for i in range(num_samples):
        genotype = dataset[i]
        print(f"\nSample {i}: Original test {genotype}")

        phenotye = gen.orig_gen.genotype2phenotype(genotype)

        if validator:
            valid_orig = validator.is_valid(phenotye)
            print(f"Original validity: {valid_orig}")
        else:
            valid_orig = "Unknown"

        gen.orig_gen.visualize_test(
            phenotye, save_path=f"{save_path}_original_images", num=i, title=valid_orig
        )

        encoded_test = gen.encode_test(genotype)

        decoded_test = gen.decode_test(encoded_test)

        print(f"Decoded test: {decoded_test}")

        decoded_phenotype = gen.orig_gen.genotype2phenotype(decoded_test)

        if validator:
            valid_decoded = validator.is_valid(decoded_phenotype)
            print(f"Decoded validity: {valid_decoded}")
        else:
            valid_decoded = "Unknown"

        gen.orig_gen.visualize_test(
            decoded_phenotype,
            save_path=f"{save_path}_vae_images",
            num=i,
            title=valid_decoded,
        )

        error = np.sum(np.square(genotype - decoded_test))
        print(f"Reconstruction error: {error:.6f}")


if __name__ == "__main__":

    module_name = "rilast.test_generators.latent_lkas_test_generator"
    class_name = "LatentLKASTestGenerator"

    # module_name ="rilast.test_generators.latent_uav_test_generator"
    # class_name = "LatentUAVTestGenerator"

    module = importlib.import_module(module_name)
    generator_class = getattr(module, class_name)

    test_generator = generator_class(save_path="temp")
    test_generator.initialize_vae()
    test_generator.initialize_problem()

    sample_dataset(
        200,
        test_generator.model,
        test_generator.archive,
        test_generator.generator,
        test_generator.config["model_path"],
        test_generator.validator,
    )
