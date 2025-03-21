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
from matplotlib import pyplot as plt
from ambiegenvae.common.vae_training_utils import calculate_latent_space, plot_latent_space_given, measure_rec_orig_similarity, verify_latent_statistics
from train_vae import VecVAE, VecVAESimple, DeepVecVAE, VecDataSet, DeepVecVAE2, Denormalize1D, Normalize1D_z, Denormalize1D_z, Denormalize1D_1, Normalize1D_1
from train_vae import loss_functionA, loss_functionB
from train_vae import train_step, test_step, save_model, load_model
from train_vae import Normalize1D, ToTensor1D
from ambiegenvae.generators.kappa_generator import KappaRoadGenerator
from sklearn.model_selection import KFold
from ambiegenvae.generators.obstacle_generator import ObstacleGenerator
from ambiegenvae.validators.obstacle_scene_validator import ObstacleSceneValidator
from aerialist.px4.obstacle import Obstacle
from ambiegenvae.validators.road_validator import RoadValidator
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

def sweep_train(config=None):

    with wandb.init(entity="dmhum",  config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        nDim = 17 # # 20#Dimension of the input vector for VAE
        nLat = 17 # #20#

        models = {"VecVAESimple": VecVAESimple(nDim, nLat), "VecVAE": VecVAE(nDim, nLat), "DeepVecVAE": DeepVecVAE(nDim, nLat), "DeepVecVAE2": DeepVecVAE2(nDim, nLat)}
        loss_functions = {"lossA": loss_functionA, "lossB": loss_functionB}

        checkpoint_freq = 500
        config = wandb.config
        print("Config", config)
        model = models[config["model"]]
        print("Model", model)
        loss_fn = loss_functions[config["loss_func"]]
        print("Loss", loss_fn)
        model = model.to(device)
        batch_size = config["batch_size"]
        learn_rate = config["lr"]

        epochs = 2001

        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        result_folder = config["name"] + config.model + "_" + config.loss_func + "_train"
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
                #transform=Denormalize1D(min, max),
                #gen=generator,
                #validator=validator

            )
            test_loss = test_step(model=model, test_loader=test_load, loss_function=loss_fn)
                                  #transform=Denormalize1D(min, max), gen=generator, validator=validator)

            #mean_l, std_l, all_latents = verify_latent_statistics(model, test_load)
            #similarity, valid_prct = measure_rec_orig_similarity(model, device, min, max, dataset_test, generator, validator, Denormalize1D_1(min,max ))
            #similarity, valid_prct = measure_rec_orig_similarity(model, device, mean, std, dataset_test, generator, validator, Denormalize1D_z(mean, std))
            #wandb.log({"similarity": similarity, "valid_prct": valid_prct})
           # wandb.log({"mean": mean_l, "std": std_l})
            wandb.log({"train_loss": train_loss, "validation_loss": test_loss, "epoch": epoch, "time":round(time.time() - start_time, 3)})
            

            if epoch % checkpoint_freq ==0:
                save_model(epoch, model, result_folder)
                #plot_latent_space_given(all_latents, epoch, result_folder + "_plots")



            # 4. Print out what's happening
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
            
def cross_validation_sweep(config=None):


    kf = KFold(n_splits=5, shuffle=True)

    for fold, (ix_train, ix_val) in enumerate(kf.split(archive)):
        print("Fold", fold)
        data_train = archive[ix_train]
        data_test = archive[ix_val]

        my_transforms = transforms.Compose([ToTensor1D(), Normalize1D(min, max)])
        dataset_train_ = VecDataSet(data_train, data_transforms=my_transforms)
        dataset_test_ = VecDataSet(data_test, data_transforms=my_transforms)

        run = wandb.init(entity="dmhum", project="AmbieGenVAE_cv_vehicle_2", config=config, group="experiment_01-04-2024",  job_type="train", reinit=True, name=f"fold_{fold}")
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        nDim = 25#17  # 20#Dimension of the input vector for VAE
        nLat = 25#17  #20#

        models = {"VecVAESimple": VecVAESimple(nDim, nLat), "VecVAE": VecVAE(nDim, nLat), "DeepVecVAE2": DeepVecVAE2(nDim, nLat), "DeepVecVAE": DeepVecVAE(nDim, nLat)}
        loss_functions = {"lossA": loss_functionA, "lossB": loss_functionB}

        checkpoint_freq = 500
        config = config#wandb.config
        print("Config", config)
        model = models[config["model"]]
        print("Model", model)
        loss_fn = loss_functionA
        print("Loss", loss_fn)
        model = model.to(device)
        batch_size = config["batch_size"]
        learn_rate = config["lr"]

        epochs = 2005

        optimizer = optim.Adam(model.parameters(), lr=learn_rate)


        result_folder = config["name"] + config["model"] + "_" + config["loss_func"] + "_train_" + str(fold)

        
        train_load = DataLoader(dataset_train_, batch_size=batch_size, shuffle=True, num_workers=4)
        test_load = DataLoader(dataset_test_, batch_size=batch_size, shuffle=False, num_workers=4)

        
        out, _ = next(iter(test_load))
        #print(out)


        results = {"train_loss": [], "test_loss": []}

        # Put model in training mode
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

            #mean_l, std_l, all_latents = verify_latent_statistics(model, test_load)
            #similarity = measure_rec_orig_similarity(model, device, mean, std, dataset_test, generator)
            #wandb.log({"similarity": similarity})
            #wandb.log({"mean": mean_l, "std": std_l})
            wandb.log({"train_loss": train_loss, "validation_loss": test_loss, "epoch": epoch})

            if epoch % checkpoint_freq ==0:
                save_model(epoch, model, result_folder)
                #plot_latent_space_given(all_latents, epoch, result_folder + "_plots")

            # Checkpoint
            #if epoch % checkpoint_freq == 0:
            #    save_model(epoch, model, result_folder)

            # 4. Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f} | "
            )

            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)

        wandb.finish()

if __name__ == "__main__":

    #archive = np.load("dataset_04_11_23_10k.npy")
    archive = np.load("dataset_07_11_23_10k_17kappa.npy") #last one, optimized
    #archive = np.load("dataset_13_11_23_10k_20kappa.npy")
    

    #archive = np.load("datasets\\01-02-2024-random_17_10k.npy"
    min = np.min(archive, axis=0)
    max = np.max(archive, axis=0)
         

        # Shuffle the data randomly
    np.random.shuffle(archive)

    # Define the split ratio (80% training, 20% validation)
    split_ratio = 0.8

    # Calculate the split index
    split_index = int(len(archive) * split_ratio)

    # Split the data into training and validation sets
    data_train = archive[:split_index]
    data_test = archive[split_index:]

    #data_train = archive[: int(0.8 * len(archive))]
    #data_test = archive[int(0.8 * len(archive)) :]


    #my_transforms = transforms.Compose([ToTensor1D(), Normalize1D_z(mean, std)]) # 
    my_transforms = transforms.Compose([ToTensor1D(), Normalize1D_1(min, max)])
    dataset_train = VecDataSet(data_train, data_transforms=my_transforms)
    dataset_test = VecDataSet(data_test, data_transforms=my_transforms)

    generator = KappaRoadGenerator(200, solution_size=17)
    validator = RoadValidator(200)

   
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
        'name': {'values': ['02-12-2024-rq4-ads_tanh_-1_1_run2']}
     }
    }

    #run_config = {'model': "VecVAESimple", 'loss_func': "lossA", 'name': "09-01-2024-cross_validation-"}
    #'batch_size': {'values': [64, 128, 512]},
    #      'lr': {'values': [0.001, 0.0001, 0.00001]}
    sweep_id = wandb.sweep(sweep_config,project='02-12-2024-rq4-ads_tanh_-1_1s', entity="dmhum")
    ## Initialize the controller and start an agent

    import time
    start_time = time.time()
    wandb.agent(sweep_id, function = sweep_train) #sweep_train
    #cross_validation_sweep(config=run_config)
    print("--- %s seconds ---" % (time.time() - start_time))
        