import numpy as np
import os
import torch
import torch
import torch.nn as nn
import sklearn.metrics as skmetrics
import timeit

from logger import get_logger


def simple_model():
    model = nn.Sequential(
        # First Conv Layer: [Fs/2, 128, //Fs/4]
        nn.Conv1d(in_channels=1, out_channels=128, kernel_size=Fs//2, stride=Fs//4, bias=False),
        nn.ReLU(inplace=True),

        # First Max-Pool Layer: [8, /8]
        nn.MaxPool1d(kernel_size=8, stride=8),

        # Dropout (only used during training)
        nn.Dropout(p=0.5),

        # Second Conv Layer: [8, 128, /1]
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False),
        nn.ReLU(inplace=True),

        # Third Conv Layer: [8, 128, /1]
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False),
        nn.ReLU(inplace=True),

        # Fourth Conv Layer: [8, 128, /1]
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False),
        nn.ReLU(inplace=True),

        # Second Max-Pool Layer: [4, /4]
        nn.MaxPool1d(kernel_size=4, stride=4),

        # Dropout (only used during training)
        nn.Dropout(p=0.5),

        # Flatten layer for fully connected input
        nn.Flatten()
    )
    return model


class SimpleModel:

    def __init__(self, config, device):
        self.model = simple_model()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.global_epoch = 0
        self.global_step = 0
        self.device = device
        self.config = config
        self.model.to(self.device)

    def train(self, minibatch_fn):
        start = timeit.default_timer()
        preds, trues, losses, outputs = ([], [], [], {})
        self.model.train()
        for x, y, w, sl, re in minibatch_fn:
            x = torch.from_numpy(x).to(self.device)  # shape(batch_size * seq_length, in_channels, input_length)
            y = torch.from_numpy(y).to(self.device)  # shape(batch_size * seq_length, )
            w = torch.from_numpy(w).to(self.device)  # shape(batch_size * seq_length, )

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            loss = torch.mul(loss, w)  # w=0 if for padded samples
            loss = loss.sum() / w.sum()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            self.global_step += 1

            tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1), (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y.cpu().detach().numpy(), (self.config["batch_size"], self.config["seq_length"]))
            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "global_step": self.global_step,
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/loss": all_loss,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,

        })
        self.global_epoch += 1
        return outputs

    def evaluate(self, minibatch_fn):
        start = timeit.default_timer()
        preds, trues, losses, outputs = ([], [], [], {})
        self.model.eval()
        with torch.no_grad():
            for x, y, w, sl, re in minibatch_fn:
                x = torch.from_numpy(x).to(self.device)  # shape(batch_size * seq_length, in_channels, input_length)
                y = torch.from_numpy(y).to(self.device)  # shape(batch_size * seq_length, )
                w = torch.from_numpy(w).to(self.device)  # shape(batch_size * seq_length, )

                y_pred = self.model(x)
                loss = self.loss(y_pred, y)
                loss = torch.mul(loss, w)  # w=0 if for padded samples
                loss = loss.sum() / w.sum()
                losses.append(loss.cpu().detach().numpy())

                tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1), (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(y.cpu().detach().numpy(), (self.config["batch_size"], self.config["seq_length"]))
                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "test/trues": trues,
            "test/preds": preds,
            "test/accuracy": acc,
            "test/loss": all_loss,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,

        })
        return outputs

    def save_checkpoint(self, name, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, "{}.ckpt".format(name))
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load_checkpoint(self, name, model_dir):
        load_path = os.path.join(model_dir, "{}.ckpt".format(name))
        self.model.load_state_dict(torch.load(load_path))
        return load_path


if __name__ == '__main__':
    model = simple_model()
    fake_x = torch.randn(size=(2, 1, 3000))
    print(f"fake_x: {fake_x.shape}")
    y_pred = model(fake_x)
    print(f"y_pred: {y_pred.shape}")
    print('Successfully run the model')
