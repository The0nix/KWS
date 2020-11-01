import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional
import wandb


import core


class Attention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        Soft attention:
            Compute energy: e_t = v^T @ tanh(W @ h_t + b)
            Compute weights: alpha = softmax(e)
            Compute outputs: out = sum(x * alpha)
        :param input_size: size of th input vectors
        :param hidden_size: size of hidden state (width of matrix W)
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor of size (batch_size, seq_len, input_size)
        :return: tensor of size (batch_size, hidden_size)
        """
        # (batch_size, seq_len, input_size)
        e = self.fc(x)                   # -> (batch_size, seq_len, hidden_size)
        e = self.v(e)                    # -> (batch_size, seq_len, 1)
        alpha = torch.softmax(e, dim=1)  # -> (batch_size, seq_len, 1)
        x = (x * alpha).sum(dim=1)       # (batch_size, seq_len, input_size) -> (batch_size, hidden_size)
        return x


class KWSNet(pl.LightningModule):
    """
    Attention-based Keyword Spotting model
    :param n_mels: number of channels in input mel-spectrograms
    :param n_keywords: number of keywords -- number of output channels
    :param cnn_channels: number of channels in CNN
    :param cnn_kernel_size: kernel size in CNN
    :param gru_hidden_size: size of the hidden state of GRU
    :param attention_hidden_size: size of the hidden state of Attention
    :param optimizer_lr: learning rate for the optimizer
    """
    def __init__(self, n_mels: int, n_keywords: int, cnn_channels: int, cnn_kernel_size: int,
                 gru_hidden_size: int, attention_hidden_size: int, optimizer_lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_lr = optimizer_lr

        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, cnn_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden_size, bidirectional=True, batch_first=True)
        self.attention = Attention(gru_hidden_size * 2, attention_hidden_size)
        self.fc = nn.Linear(gru_hidden_size * 2, n_keywords + 1, bias=False)

        self.fafncurve = core.metrics.FAFNCurve(compute_on_step=False)

    def forward(self, x):
        """
        :param x: tensor of size (batch_size, n_mels, seq_len)
        :return: tensor of probability distributions of size (batch_size, n_keywords + 1)
        """
        # (batch_size, n_mels, seq_len)
        x = self.cnn(x)         # -> (batch_size, cnn_channels, seq_len)
        x = x.permute(0, 2, 1)  # -> (batch_size, seq_len, cnn_channels)
        x, _ = self.rnn(x)      # -> (batch_size, seq_len, gru_hidden_size * 2)
        x = self.attention(x)   # -> (batch_size, gru_hidden_size * 2)
        x = self.fc(x)          # -> (batch_size, n_keywords)
        x = torch.log_softmax(x, dim=1)
        return x

    def inference(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Perform inference with window of given size
        :param x: torch.tensor of shape (batch_size, n_mels, seq_len) or (n_mels, seq_len)
        :param window_size: size of the window to pass through the tensor with
        :return: tensor p of size (batch_size, n_keywords, seq_len - window_size + 1)
        with probability of each keyword for each window
        """
        assert len(x.shape) in {2, 3}
        if len(x.shape) == 2:
            x.unsqueeze(0)
        # (batch_size, n_mels, seq_len)
        x = self.cnn(x)         # -> (batch_size, cnn_channels, seq_len)
        x = x.permute(0, 2, 1)  # -> (batch_size, seq_len, cnn_channels)
        x, _ = self.rnn(x)      # -> (batch_size, seq_len, gru_hidden_size * 2)

        if window_size > x.shape[1]:
            # If the size of audio is less then the window, we will only do one pass
            window_size = x.shape[1]
        ps = []  # (n_windows, batch_size, n_keywords)
        for i in range(window_size, x.shape[1] + 1):
            window = x[:, i - window_size:i]
            window = self.attention(window)    # -> (batch_size, gru_hidden_size * 2)
            window = self.fc(window)           # -> (batch_size, n_keywords + 1)
            p = torch.softmax(window, dim=1)  # -> (batch_size, n_keywords + 1)
            ps.append(p)

        # (n_windows, batch_size, n_keywords + 1) -> (batch_size, n_keywords + 1, n_windows)
        ps = torch.stack(ps).permute(1, 2, 0)
        return ps

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        spectrograms, lenghts, keyword_ids = batch
        probas = self(spectrograms.permute(0, 2, 1))
        loss = nn.NLLLoss()(probas, keyword_ids)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        spectrograms, lenghts, keyword_ids = batch
        probas = self(spectrograms.permute(0, 2, 1))
        loss = nn.NLLLoss()(probas, keyword_ids)

        self.fafncurve.forward(probas, keyword_ids)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        FA, FR, _ = self.fafncurve.compute()
        self.fafncurve.reset()
        data = list(zip(FA, FR))
        table = wandb.Table(data=data, columns=["x", "y"])
        self.logger.experiment.log({"Logging": wandb.plot.line(table, "x", "y", title="FA/FR curve")})
        self.log("au_fa_fr", pl.metrics.functional.classification.auc(FA, FR))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_lr)
        return optimizer
