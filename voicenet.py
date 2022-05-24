import os
from tqdm import tqdm

import numpy as np

import torch
import torchvision

from tensorboardX import SummaryWriter
from typing import Optional

from voiceds import *

DEBUG = False
if DEBUG:
    workers = 0
    pin_memory = False
    persistent_workers = False
else:
    workers = 4
    pin_memory = True
    persistent_workers = True


class VoiceClassifier:
    def __init__(self, epochs=5, device='cuda:0', checkpoint: Optional[str] = None):
        self.net = torchvision.models.resnet50()

        self.device = device
        # Chose because of a large
        self.net = torchvision.models.resnet50(num_classes=1)
        # Modify fast since a spectrogram is 1-D image
        self.net.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.net.to(device)
        self.epochs = epochs
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adagrad(self.net.parameters(), lr=0.01, weight_decay=0.9)

        self.ds = VoiceDataset(batch=64, subset='train_train', normalize=True)
        self.dloader = DataLoader(
            self.ds,
            batch_size=self.ds.batch,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

        if checkpoint is None:
            pass
            print('Initialized from zero')
        else:
            assert os.path.exists(checkpoint), checkpoint
            self.net.load_state_dict(torch.load(checkpoint))

    def train(self):
        global_step = 0
        short = tlog().replace(":", "")
        logdir = f'models/model-{short}'
        print('Log -> ', logdir)
        tXw = SummaryWriter(logdir)
        print('Initialized tXw')

        for epoch in range(-1, self.epochs):
            tenth = len(self.dloader) // 10
            loss_sum = 0
            acc_sum = 0
            self.net.train()

            for i, sample in tqdm(enumerate(self.dloader), desc='Train loop', total=len(self.dloader)):
                imgs = sample['img'].to(self.device)
                labels = sample['label'].to(self.device)

                self.opt.zero_grad()
                predictions = self.net.forward(imgs).reshape(-1)

                cur_loss = self.loss(
                    input=predictions,
                    target=labels.float()
                )
                cur_loss.backward()
                self.opt.step()

                # Log
                loss_sum += cur_loss.item()
                pred_labels_raw = torch.sigmoid(predictions)
                pred_labels = torch.round(pred_labels_raw)

                # print(cats, pred_labels)
                acc = (pred_labels == labels).detach().cpu().numpy().mean()
                acc_sum += acc

                tXw.add_scalar('Loss/Train', cur_loss.item(), global_step)
                tXw.add_scalar('Acc/Train', acc, global_step)
                if epoch == -1:
                    break
                elif i % tenth == 0 and i > 0:
                    print(
                        f'Epoch {epoch:03} '
                        f'Step {i:03} '
                        f'Loss mean {loss_sum / i:.4f} '
                        f'Acc. mean {acc_sum / i:.4f} '
                    )
                global_step += 1

            i = len(self.ds)
            print(
                f'E{epoch:03} '
                f'Step {i:03} '
                f'Loss mean {loss_sum / i:.4f} '
                f'Acc. mean {acc_sum / i:.4f} '
            )
            acc_val = self.eval(subset='train_train')
            tXw.add_scalar('Acc/TrainEpoch', acc_val, global_step)
            acc_val = self.eval(subset='train_val')
            tXw.add_scalar('Acc/Val', acc_val, global_step)

        acc_test = self.eval(subset='val')
        tXw.add_scalar('Acc/Test', acc_test, global_step)
        print(f'End of training. Test acc. {acc_test:.4f}')

    def eval(self, subset):
        self.net.eval()

        ds = VoiceDataset(batch=1, subset=subset, normalize=True)
        dloader = DataLoader(
            ds,
            batch_size=ds.batch,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

        labels_true_all = []
        labels_pred_all = []
        print(tlog(), 'Iterating over dataloader of length ', len(ds))

        for i, sample in tqdm(enumerate(dloader), desc='Evaluation loop', total=len(dloader)):
            imgs = sample['img'].to(self.device)
            labels = sample['label'].to(self.device)

            with torch.no_grad():
                predictions = self.net.forward(imgs)
            pred_labels_raw = torch.sigmoid(predictions)
            pred_labels = torch.round(pred_labels_raw)

            # print(cats, pred_labels)
            labels_true_all.extend(list(labels.cpu().numpy()))
            labels_pred_all.extend(list(pred_labels.detach().cpu().numpy()))

        labels_true_all = np.array(labels_true_all)
        labels_pred_all = np.array(labels_pred_all)

        acc = np.mean(labels_pred_all == labels_true_all)
        print(tlog(), f'Evaluation accuracy: {acc:.4f}')
        return acc


if __name__ == '__main__':
    vcls = VoiceClassifier()
    vcls.train()
