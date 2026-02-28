import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as T
import visdom
from visdom.loggers.lightning_logger import LightningGradientNormLogger
from visdom.loggers.lightning_profiler import LightningHookProfiler, LightningOpProfiler


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class LitMnist(pl.LightningModule):
    def __init__(self, vis, env, lr=1e-3):
        super().__init__()
        self.model = MnistCNN()
        self.vis   = vis    # visdom connection for logging loss and accuracy
        self.env   = env    # visdom environment name
        self.lr    = lr     # learning rate passed to Adam

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out  = self(x)
        loss = nn.CrossEntropyLoss()(out, y)
        acc  = out.detach().argmax(1).eq(y).float().mean() * 100  # batch accuracy in percent
        self.vis.line(Y=[loss.item()], X=[self.global_step], win="train_loss",
                      env=self.env, update="append", opts={"title": "Train Loss",
                                                           "xlabel": "Step", "ylabel": "Loss"})
        self.vis.line(Y=[acc.item()],  X=[self.global_step], win="train_acc",
                      env=self.env, update="append", opts={"title": "Train Accuracy",
                                                           "xlabel": "Step", "ylabel": "Accuracy (%)"})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        acc  = self(x).argmax(1).eq(y).float().mean() * 100
        self.log("val_acc", acc)  # lightning stores this so we can read it in the epoch end hook

    def on_validation_epoch_end(self):
        val_acc = self.trainer.callback_metrics.get("val_acc")
        if val_acc is not None:
            self.vis.line(Y=[val_acc.item()], X=[self.current_epoch], win="val_acc",
                          env=self.env, update="append", opts={"title": "Val Accuracy",
                                                               "xlabel": "Epoch", "ylabel": "Accuracy (%)"})

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class MnistData(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128):
        super().__init__()
        self.data_dir   = data_dir
        self.batch_size = batch_size
        self.tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download happens here (called only on rank 0 in multi-GPU setups)
        torchvision.datasets.MNIST(self.data_dir, train=True,  download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_ds = torchvision.datasets.MNIST(self.data_dir, train=True,  transform=self.tf)
        self.val_ds   = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.tf)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=256, shuffle=False, num_workers=2
        )


def train(args):
    vis = visdom.Visdom(port=args.vis_port, env=args.env)

    # create profilers only if the user asked for them
    hook_prof = LightningHookProfiler() if args.hook_profile else None
    op_prof   = LightningOpProfiler()   if args.op_profile   else None

    grad_cb = LightningGradientNormLogger(
        vis,
        log_every=args.log_every,
        env=args.env,
        hook_profiler=hook_prof,
        op_profiler=op_prof,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[grad_cb],   # our callback handles hook attach/detach automatically
        logger=False,
    )
    trainer.fit(LitMnist(vis, args.env), MnistData(args.data_dir, args.batch_size))

    if hook_prof:
        print("\nLightningHookProfiler summary:", hook_prof.summary())
    if op_prof:
        print("\nLightningOpProfiler top ops:\n", op_prof.summary(top_n=10))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int, default=3)
    p.add_argument("--batch-size",   type=int, default=128, dest="batch_size")
    p.add_argument("--log-every",    type=int, default=10,  dest="log_every")
    p.add_argument("--vis-port",     type=int, default=8097, dest="vis_port")
    p.add_argument("--env",          type=str, default="lightning_mnist")
    p.add_argument("--data-dir",     type=str, default="./data", dest="data_dir")
    p.add_argument("--hook-profile", action="store_true", dest="hook_profile")
    p.add_argument("--op-profile",   action="store_true", dest="op_profile")
    train(p.parse_args())