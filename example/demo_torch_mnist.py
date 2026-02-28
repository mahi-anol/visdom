import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import visdom
from visdom.loggers.torch_logger import GradientNormLogger
from visdom.loggers.torch_profiler import HookProfiler, TorchOpProfiler


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


def get_loaders(batch_size, data_dir):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=2)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            total   += y.size(0)
    return correct / total * 100


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis = visdom.Visdom(port=args.vis_port, env=args.env)

    train_loader, test_loader = get_loaders(args.batch_size, args.data_dir)

    model = MnistCNN().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    # create profilers only if the user asked for them
    hook_prof = HookProfiler()    if args.hook_profile else None
    op_prof   = TorchOpProfiler() if args.op_profile   else None

    logger = GradientNormLogger(
        vis, model,
        log_every=args.log_every,
        env=args.env,
        hook_profiler=hook_prof,
        op_profiler=op_prof,
    )
    logger.attach()

    step = 0  # global step counter across all epochs

    if op_prof is not None:
        # TorchOpProfiler is a context manager, so wrap the whole loop in it
        op_prof.__enter__()

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = total = 0  # track accuracy within this epoch

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out  = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            logger.step()  # records norms and advances profiler schedule
            opt.step()

            step    += 1
            correct += out.detach().argmax(1).eq(y).sum().item()
            total   += y.size(0)

            if step % args.log_every == 0:
                vis.line(Y=[loss.item()],           X=[step], win="train_loss", env=args.env,
                         update="append", opts={"title": "Train Loss",
                                                "xlabel": "Step", "ylabel": "Loss"})
                vis.line(Y=[correct / total * 100], X=[step], win="train_acc",  env=args.env,
                         update="append", opts={"title": "Train Accuracy",
                                                "xlabel": "Step", "ylabel": "Accuracy (%)"})

        val_acc = evaluate(model, test_loader, device)
        vis.line(Y=[val_acc], X=[epoch], win="val_acc", env=args.env,
                 update="append", opts={"title": "Val Accuracy",
                                        "xlabel": "Epoch", "ylabel": "Accuracy (%)"})
        print(f"epoch {epoch} | val acc {val_acc:.2f}%")

    if op_prof is not None:
        op_prof.__exit__(None, None, None)

    logger.detach()

    if hook_prof:
        print("\nHookProfiler summary:", hook_prof.summary())
    if op_prof:
        print("\nTorchOpProfiler top ops:\n", op_prof.summary(top_n=10))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int, default=3)
    p.add_argument("--batch-size",   type=int, default=128, dest="batch_size")
    p.add_argument("--log-every",    type=int, default=10,  dest="log_every")
    p.add_argument("--vis-port",     type=int, default=8097, dest="vis_port")
    p.add_argument("--env",          type=str, default="torch_mnist")
    p.add_argument("--data-dir",     type=str, default="./data", dest="data_dir")
    p.add_argument("--hook-profile", action="store_true", dest="hook_profile")
    p.add_argument("--op-profile",   action="store_true", dest="op_profile")
    train(p.parse_args())