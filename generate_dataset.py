import time
import importlib
import numpy as np
import os
import json
from datetime import datetime
from rilast.common.parse_arguments import parse_arguments_dataset_generation
from rilast.generators.abstract_generator import AbstractGenerator


def generate_dataset(
    generator: AbstractGenerator,
    samples: int,
    save_dir: str = "./",
    save_name: str = "result",
) -> np.ndarray:
    """Generate a dataset of samples from a generator.

    Args:
        generator (AbstractGenerator): Generator to use.
        samples (int): Number of samples to generate.
        save_dir (str): Directory to save the dataset.
        save_name (str): Name of the saved dataset file.

    Returns:
        np.ndarray: Generated dataset.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start = time.time()
    dataset = np.zeros((samples, generator.size))
    tc_count = 0
    while tc_count < samples:
        test, valid = generator.generate_random_test()
        test = np.array(generator.genotype)
        if valid:
            dataset[tc_count] = test
            tc_count += 1

    file_name = os.path.join(save_dir, save_name + ".npy")
    np.save(file_name, dataset)
    print(f"Dataset saved in {file_name}")
    print(f"Dataset generation time: {time.time() - start:.2f} seconds")

    return dataset

def generate_dataset_from_folder(
    generator: AbstractGenerator,
    tc_file_path: str,
    num_samples: int = 10000,
    save_dir: str = "./",
    save_name: str = "result",
) -> np.ndarray:
    """Generate a dataset of samples from a generator.

    Args:
        generator (AbstractGenerator): Generator to use.
        tc_file_path (str): Path to the test case file.
        num_samples (int): Number of samples to generate.
        save_dir (str): Directory to save the dataset.
        save_name (str): Name of the saved dataset file.

    Returns:
        np.ndarray: Generated dataset.
    """
    start = time.time()
    dataset = np.zeros((num_samples, generator.phenotype_size))

    with open(tc_file_path, "r") as f:
        all_scenarios = json.load(f)

    tc_count = 0
    for run in all_scenarios:
        scenarios = all_scenarios[run]
        for scenario in scenarios:
            dataset[tc_count] = np.array(scenarios[scenario])
            tc_count += 1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Dataset size: {tc_count}")
    file_name = os.path.join(save_dir, save_name + ".npy")
    np.save(file_name, dataset)
    print(f"Dataset saved in {file_name}")
    print(f"Dataset generation time: {time.time() - start:.2f} seconds")

    return dataset

if __name__ == "__main__":
    args = parse_arguments_dataset_generation()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")

    module_name = args.module_name
    class_name = args.class_name
    dataset_dir = args.dataset_dir or "datasets"
    dataset_size = args.size
    test_case_dir = args.tc_dir

    module = importlib.import_module(module_name)
    generator_class= getattr(module, class_name)

    test_generator = generator_class(save_path="temp")
    test_generator.initialize_problem()

    generator = test_generator.generator

    

    if test_case_dir is None:
        dataset_name = args.dataset_name or f"{dt_string}_{class_name}_{dataset_size}_random"
        print("Starting random dataset generation")
        generate_dataset(
            generator, dataset_size, save_dir=dataset_dir, save_name=dataset_name
        )
    else:
        dataset_name = args.dataset_name or f"{dt_string}_{class_name}_optimized"
        print("Starting optimized dataset generation")
        generate_dataset_from_folder(
            generator,
            test_case_dir,
            num_samples=dataset_size,
            save_dir=dataset_dir,
            save_name=dataset_name,
        )
