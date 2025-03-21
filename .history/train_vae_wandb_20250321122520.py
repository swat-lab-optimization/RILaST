import sys
import argparse
import torch
import math
import os
import wandb
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import numpy as np
import time
from matplotlib import pyplot as plt

from rilast.common import train_step, test_step, save_model, load_model
from rilast.common.vae_training_utils import VecDataSet, Normalize1D_1, ToTensor1D, loss_functionA, Denormalize1D_1
from rilast.common import VAE


def sweep_train(config=None):

    with wandb.init(entity="dmhum",  config=config):
        config = wandb.config
        print("Config", config)
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        nDim = 17 # # 20#Dimension of the input vector for VAE
        nLat = config["nlat"]# #20#

        models = {"VecVAESimple": VecVAESimple(nDim, nLat), "VecVAE": VecVAE(nDim, nLat), "DeepVecVAE": DeepVecVAE(nDim, nLat), "DeepVecVAE2": DeepVecVAE2(nDim, nLat)}
        loss_functions = {"lossA": loss_functionA, "lossB": loss_functionB}

        checkpoint_freq = 500

        model = models[config["model"]]
        print("Model", model)
        loss_fn = loss_functions[config["loss_func"]]
        print("Loss", loss_fn)
        model = model.to(device)
        batch_size = config["batch_size"]
        learn_rate = config["lr"]

        epochs = 1001

        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        result_folder = config["name"] + config.model + "_" + config.loss_func + "_" + str(config.nlat) + "_train"
        train_load = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        test_load = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        out, _ = next(iter(test_load))
        #print(out)
        results = {"train_loss": [], "test_loss": []}

        # Put model in training mode
        start_time = time.time()
        model.train()
        # 3. Loop through training and testing steps for a number of epochs
        for epoch in (range(epochs)):
            train_loss = train_step(
                model=model,
                train_loader=train_load,
                loss_function=loss_fn,
                optimizer=optimizer

            )
            test_loss = test_step(model=model, test_loader=test_load, loss_function=loss_fn)

            wandb.log({"train_loss": train_loss, "validation_loss": test_loss, "epoch": epoch, "time":round(time.time() - start_time, 3)})
            

            if epoch % checkpoint_freq ==0:
                save_model(epoch, model, result_folder)

            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"time : {(start_time - time.time()):.4f}"
            )

            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)
            #wandb.log(results)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script to train a VAE")

    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of epochs to train (default: 1000)",
    )
    parser.add_argument(
        "--ndim",
        type=int,
        default=17,
        help="Input dimension of the VAE (default: 17)",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets\\dataset_random_ads.npy",
        help="data",
    )

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")


    nDim = args.ndim  # 29 Dimension of the input vector for VAE

    # Load data set
    archive = np.load(args.data_path)
    np.random.shuffle(archive)
    data_train = archive[: int(0.8 * len(archive))]
    data_test = archive[int(0.8 * len(archive)) :]

    min = np.min(archive, axis=0)  # min for each feature
    max = np.max(archive, axis=0)

    # Normalize data
    my_transforms = transforms.Compose([ToTensor1D(), Normalize1D_1(min, max)])
    dataset_train = VecDataSet(data_train, data_transforms=my_transforms)
    dataset_test = VecDataSet(data_test, data_transforms=my_transforms)

   
    wandb.login()

    sweep_config = {
    'method': 'grid',
    'name': '02-12-2024-rq4-ads_tanh_-1_1_run2',#'29-01-2024-non-random-ads_tanh_-1_1',
    'metric': {
        'goal': 'minimize', 
        'name': 'validation_loss'
        },
    'parameters': {
         'model': {'values': ["DeepVecVAE", "VecVAESimple", "VecVAE"]}, #"VecVAESimple", "VecVAE",
        'loss_func': {'values': ["lossA"]} ,#, "lossB"
        'batch_size': {'values': [64, 128, 512]},
        'lr': {'values': [0.001, 0.0001]},
        'nlat': {'values': [17]},
        'name': {'values': ['02-12-2024-rq4-ads_tanh_-1_1_run2']}
     }
    }

    #run_config = {'model': "VecVAESimple", 'loss_func': "lossA", 'name': "09-01-2024-cross_validation-"}
    #'batch_size': {'values': [64, 128, 512]},
    #      'lr': {'values': [0.001, 0.0001, 0.00001]}
    
    ## Initialize the controller and start an agent

    start_time = time.time()

    for run in range(5):
        sweep_id = wandb.sweep(sweep_config,project='31-12-2024-rq3-ads_tanh-2', entity="dmhum")
        #current_run = run
        np.random.shuffle(archive)

    # Define the split ratio (80% training, 20% validation)
        split_ratio = 0.8
        # Calculate the split index
        split_index = int(len(archive) * split_ratio)

        # Split the data into training and validation sets
        data_train = archive[:split_index]
        data_test = archive[split_index:]

        #my_transforms = transforms.Compose([ToTensor1D(), Normalize1D_z(mean, std)]) # 
        my_transforms = transforms.Compose([ToTensor1D(), Normalize1D_1(min, max)])
        dataset_train = VecDataSet(data_train, data_transforms=my_transforms)
        dataset_test = VecDataSet(data_test, data_transforms=my_transforms)
        wandb.agent(sweep_id, function = sweep_train) #sweep_train
        #cross_validation_sweep(config=run_config)
        print("--- %s seconds ---" % (time.time() - start_time))
        