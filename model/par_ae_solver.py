import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from tqdm import tqdm, trange

from layers import ParallelAutoencoder
from utils import TensorboardWriter

# NEED TO BE TAKEN TO THE CONFIG FILE
INPUT_SIZE = 1024
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
LEARNING_RATE = 1e-4
LOG_DIR = f"exp0/TVSum/par-ae/logs/split0/h{str(HIDDEN_SIZE)}_lr{str(LEARNING_RATE)}"
SAVE_MODEL_DIR = "exp0/TVSum/par-ae/model/split0/"

class ParallelAutoencoderSolver:
    """Handler for the training of the parallel autoencoder"""
    def __init__(self, train_loader=None, test_loader=None, config=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.criterion = nn.MSELoss(reduction='sum')

    def build(self):
        self.ae = ParallelAutoencoder(input_dim=INPUT_SIZE,
                                      num_hidden=HIDDEN_SIZE,
                                      num_layers=NUM_LAYERS).cuda()
        self.model = nn.ModuleList([self.ae])

        # TODO: Insert training mode from config
        self.ae_optimizer = optim.Adam(
            list(self.ae.enc.parameters()) +
            list(self.ae.dec.parameters()),
            lr=LEARNING_RATE
        )
        self.writer = TensorboardWriter(LOG_DIR)

    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc="Epoch", ncols=80):
            loss_history = []
            for batch_i, image_features in enumerate(tqdm(
                self.train_loader, desc='Batch', leave=False
            )):
                self.model.train()
                # [seq_len, 1024]
                image_features = image_features.view(-1, INPUT_SIZE)

                # [seq_len, 1024]
                image_features_ = Variable(image_features).cuda()

                # TODO: Insert into config file
                tqdm.write('\nTraining the encoder-decoder')

                # [seq_len, 1, 1024]
                decoded_features, _ = self.ae(image_features_.detach().unsqueeze(1)) # Maybe get rid of the cuda?
                decoded_features = decoded_features.cuda()

                # Calculate the loss
                loss = self.criterion(image_features_.unsqueeze(1), decoded_features)
                tqdm.write(f'\nAutoencoder loss: {loss.item():.3f}')

                # Weight update
                self.ae_optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.ae_optimizer.step()

                # Append to loss history
                loss_history.append(loss.data)

                self.writer.update_loss(loss, step, 'ae_loss')
                step += 1

            loss = torch.stack(loss_history).mean()
            self.writer.update_loss(loss, epoch_i, 'Autoencoder loss per epoch')
            chckpt_path = SAVE_MODEL_DIR + f'epoch-{epoch_i}.pkl'
            tqdm.write(f'Save parameters at {chckpt_path}')
            torch.save(self.model.state_dict(), chckpt_path)

if __name__ == "__main__":
    pass