# Core Code borrowed from https://github.com/musikalkemist/pytorchforaudio/tree/main/09%20Training%20urban%20sound%20classifier

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import accuracy_score

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

TRAIN_ANNOTATIONS_FILE = "/scratch/ys5hd/Riffusion/music/UrbanSound8K/metadata/UrbanSound8K_train.csv"
TEST_ANNOTATIONS_FILE = "/scratch/ys5hd/Riffusion/music/UrbanSound8K/metadata/UrbanSound8K_test.csv"
AUDIO_DIR = "/scratch/ys5hd/Riffusion/music/UrbanSound8K/audio/"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

BEST_ACC = -1
BEST_EPOCH = -1

# Fix Seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")
    
def test(model, data_loader, device, epoch):
    global BEST_ACC
    global BEST_EPOCH
    target_tracker = np.array([])
    prediction_tracker = np.array([])
    for input, target in data_loader:
        input = input.to(device)
        target_tracker = np.concatenate([target_tracker, target.numpy()])
        
        # calculate loss
        prediction = model(input)
        _, pred_labels = torch.max(prediction, 1)
        prediction_tracker = np.concatenate([prediction_tracker, pred_labels.detach().cpu().numpy()])                
    
    print("Accuracy: {}".format(sum(target_tracker == prediction_tracker)/len(target_tracker)))
    if BEST_ACC < sum(target_tracker == prediction_tracker)/len(target_tracker):
        BEST_ACC = sum(target_tracker == prediction_tracker)/len(target_tracker)
        BEST_EPOCH = epoch
          
def train(model, train_data_loader, test_data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        model.train()
        train_single_epoch(model, train_data_loader, loss_fn, optimiser, device)
        print("---------------------------")
        model.eval()
        test(model, test_dataloader, device, i)
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(TRAIN_ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    usd = UrbanSoundDataset(TEST_ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")
    
    test_dataloader = create_data_loader(usd, BATCH_SIZE)    
    
    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, test_dataloader, loss_fn, optimiser, device, EPOCHS)    
    
    print('Best Accuracy: {}'.format(BEST_ACC))
    
    with open('performance.txt', 'w') as f:
        f.write('Train File: {} \n'.format(TRAIN_ANNOTATIONS_FILE))
        f.write('Accuracy: {} Epoch: {} \n'.format(BEST_ACC, BEST_EPOCH))                    