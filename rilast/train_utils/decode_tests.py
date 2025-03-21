import json
from aerialist.px4.obstacle import Obstacle
from rilast.generators.obstacle_generator import ObstacleGenerator
import numpy as np
import os

from rilast.generators.latent_generator import LatentGenerator
from train_vae import (
    Denormalize1D,
    VecVAE,
    DeepVecVAE,
    DeepVecVAE2,
    VecVAESimple,
    Normalize1D,
    ToTensor1D,
    Denormalize1D_z,
    Normalize1D_z,
)
from train_vae import Denormalize1D_1, Normalize1D_1

import torch
from test_vae import load_model
from torchvision import datasets, transforms


def decode_tests(filepath, generator, base_dir):
    with open(filepath) as f:
        all_tests = json.load(f)

        for run in all_tests:
            for tc in all_tests[run]:
                if all_tests[run][tc]["info"] == "Valid test":
                    test = all_tests[run][tc]["test"]
                    decoded_test = generator.decode_test(test)
                    new_test = [float(i) for i in list(decoded_test)]

                    denormilized_test = generator.orig_gen.denormilize_flattened_test(
                        decoded_test
                    )
                    all_tests[run][tc]["original_test"] = list(denormilized_test)
                    num_obst = round(denormilized_test[0])
                    new_test = new_test[: num_obst * 6 + 1]
                    while len(new_test) < 19:
                        new_test.append(0)

                    all_tests[run][tc]["test"] = new_test

    with open(os.path.join(base_dir, "decoded_tests.json"), "w") as f:
        json.dump(all_tests, f, indent=4)


if __name__ == "__main__":

    # filepath = "ALL_RESULTS_21_Sept_24\\RQ1\\UAV\\10-04-2024_stats_ga_sbx_pm_3ob_latent_optimized\\10-04-2024-all_tests.json"
    # filepath = "ALL_RESULTS_21_Sept_24\\RQ2\\UAV\\latent-novelty\\all_pc2\\all_tests_merged.json"
    # filepath = "ALL_RESULTS_21_Sept_24\\RQ2\\UAV\\latent-random\\19-03-2024_stats_ga_sbx_pm_3ob_latent_random\\19-03-2024-all_tests.json"
    # filepath = "ALL_RESULTS_21_Sept_24\\RQ3\\UAV\\19-03-2024_stats_ga_one_point_ob_latent_obstacle_3ob_latent_random\\19-03-2024-all_tests.json"

    # filepath = "ALL_RESULTS_21_Sept_24\\RQ3\\UAV\\19-03-2024_stats_ga_sbx_pm_3ob_latent_random\\19-03-2024-all_tests.json"
    # filepath = "RESULTS_19-12_24_andre_pc_no_duplicate\\latent\\merged\\all_tests.json"
    # filepath = "latent_optimized_sbx-19-02-24-data_my_pc_duplicate_0.01\\merged\\all_tests.json"
    # filepath = "DRONE_RESULTS\\random_dataset_vector_op-30-12-24\\merged\\all_tests_norm_fin.json"
    filepath = "ALL_RESULTS_21_Sept_24\\RQ2-2\\UAV\\latent-random\\19-03-2024_stats_random_one_point_ob_latent_obstacle_3ob_latent_random\\19-03-2024-all_tests.json"
    base_dir = os.path.dirname(filepath)
    print(base_dir)

    # archive = np.load("datasets\\dataset_uav_20_03_24_10k_3ob.npy")
    # archive = np.load("final_datasets\\uav\\novelty\\dataset_uav_15_10_24_10k_3ob.npy")
    archive = np.load("final_datasets\\uav\\random\\19-03-2024-random_uav_10k.npy")
    # archive = np.load("datasets\\dataset_uav_19_02_24_10k_3ob.npy")
    mean = np.mean(archive, axis=0)  # mean for each feature
    std = np.std(archive, axis=0)
    min = np.min(archive, axis=0)
    max = np.max(archive, axis=0)

    nDim = 19  # 25
    nLat = 19  # 25

    transform = Denormalize1D_1(min, max)
    transform_norm = transforms.Compose([ToTensor1D(), Normalize1D_1(min, max)])
    model = DeepVecVAE(nDim, nLat)
    # model = load_model(1500, model, path="19-02-2024-non-random-uav_-1_1DeepVecVAE_lossA_train")#
    # self.model = load_model(1500, self.model, path="20-02-2024-random-uav_-1_1DeepVecVAE2_lossA_train") #"19-03-2024-random-uav2_-1_1DeepVecVAE_lossA_train"
    # self.model = load_model(3500, self.model, path="19-03-2024-random-uav2_-1_1DeepVecVAE_lossA_train")
    # model = load_model(3000, model, path="21-03-2024-latent-uav2_-1_1DeepVecVAE_lossA_train")
    # model = load_model(3000, model, path="15-10-2024-novelty-uav_-1_1DeepVecVAE_lossA_train")
    model = load_model(
        3000,
        model,
        path="final_models\\uav\\random\\19-03-2024-random-uav2_-1_1DeepVecVAE_lossA_train",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    min_size = Obstacle.Size(2, 2, 15)
    max_size = Obstacle.Size(20, 20, 25)
    min_position = Obstacle.Position(-40, 10, 0, 0)
    max_position = Obstacle.Position(30, 40, 0, 90)

    case_study = "case_studies/mission1.yaml"  # "case_studies/mission1.yaml"

    # Set up the algortihm

    generator = ObstacleGenerator(
        min_size,
        max_size,
        min_position,
        max_position,
        case_study_file=case_study,
        max_box_num=3,
    )

    latent_generator = LatentGenerator(
        nLat, 0, 1, generator, model, transform, transform_norm
    )

    decode_tests(filepath, latent_generator, base_dir)
