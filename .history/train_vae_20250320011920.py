import sys
import argparse
import torch
from datetime import datetime
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import numpy as np

from rilast.common import train_step, test_step, save_model, load_model
from rilast.common.vae_training_utils import VecDataSet, Normalize1D_1, ToTensor1D, loss_functionA, Denormalize1D_1
from rilast.common import VAE


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_function,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    transform: transforms.Compose = None,
    result_folder: str = None
):
    """
    Train a PyTorch model for a number of epochs.
    """

    results = {"train_loss": [], "test_loss": []}

    # Put model in training mode
    model.train()
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(
            model=model,
            train_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            transform=transform
        )
        test_loss = test_step(
            model=model,
            test_loader=test_loader,
            loss_function=loss_function,
            transform=transform
        )  # , loss_function=loss_fn)

        checkpoint_freq = 500

        # Checkpoint
        if epoch % checkpoint_freq == 0:
            save_model(epoch, model, result_folder)

        # 4. Print out what's happening
        tqdm.write(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    # 6. Return the filled results at the end of the epochs
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to train a VAE")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--learn-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--VAE-type",
        type=str,
        default="VAE3",
        help="Deafult VAE type (default: VAE3)",
    )
    
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
        "--ldim",
        type=int,
        default=17,
        help="Latent dimension of the VAE (default: 17)",
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
    nLat = args.ldim  # 29 Dimension of the latent space for VAE

    # Load data set
    archive = np.load(args.data_path)
    np.random.shuffle(archive)
    data_train = archive[: int(0.8 * len(archive))]
    data_test = archive[int(0.8 * len(archive)) :]
    batach_size = args.batch_size
    lrate = args.learn_rate

    min = np.min(archive, axis=0)  # min for each feature
    max = np.max(archive, axis=0)

    # Normalize data
    my_transforms = transforms.Compose([ToTensor1D(), Normalize1D_1(min, max)])
    dataset_train = VecDataSet(data_train, data_transforms=my_transforms)
    dataset_test = VecDataSet(data_test, data_transforms=my_transforms)

    train_load = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_load = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)


    model = VAE[args.VAE_type](nDim, nLat).to(device)
    lr = lrate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = loss_functionA
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M")
    result_folder = f"{dt_string}_{args.VAE_type}_lr_{lr}_batch_{batach_size}"
    results = train(
        model=model,
        train_loader=train_load,
        test_loader=train_load,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=args.epochs,
        transform=Denormalize1D_1(min, max),
        result_folder=result_folder
    )

    # dataset = model.dataLoader(pop)
