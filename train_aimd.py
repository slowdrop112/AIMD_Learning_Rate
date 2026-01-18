import os
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import optuna

# -------------------------
# 1. Model: Simple CNN (CIFAR-10)
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------------
# 2. AIMD Scheduler
# -------------------------
class AIMDScheduler:
    def __init__(self, optimizer, lr_init, add_amount, mult_factor, min_lr=1e-6, max_lr=0.5):
        self.optimizer = optimizer
        self.lr = lr_init
        self.add_amount = add_amount
        self.mult_factor = mult_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.prev_loss = None

        # Set initial LR
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def step(self, loss):
        if self.prev_loss is None:
            self.prev_loss = loss
            return self.lr

        # Logic: AIMD (Additive Increase, Multiplicative Decrease)
        if loss <= self.prev_loss:
            # Loss scade (sau stagnează) -> Creștem LR (Additive Increase)
            self.lr = min(self.max_lr, self.lr + self.add_amount)
        else:
            # Loss crește -> Tăiem LR (Multiplicative Decrease)
            self.lr = max(self.min_lr, self.lr * self.mult_factor)

        # Apply new LR
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

        self.prev_loss = loss
        return self.lr

# -------------------------
# 3. Data Preparation
# -------------------------
def prepare_data(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# -------------------------
# 4. Train / Eval Functions
# -------------------------
def train_epoch(model, device, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total

def eval_epoch(model, device, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += x.size(0)

    return loss_sum / total, correct / total

# -------------------------
# 5. Optuna Objective (Optimized with Pruning)
# -------------------------
def optuna_objective(trial, args, trainloader, testloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hiperparametri de căutat
    lr_init = trial.suggest_float("lr_init", 1e-4, 1e-1, log=True)
    add_amount = trial.suggest_float("add_amount", 1e-5, 1e-3, log=True)
    mult_factor = trial.suggest_float("mult_factor", 0.3, 0.9)

    # Inițializare model mic pentru viteză
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
    scheduler = AIMDScheduler(optimizer, lr_init, add_amount, mult_factor)
    criterion = nn.CrossEntropyLoss()

    # Rulăm doar 5 epoci pentru a testa rapid calitatea parametrilor
    for step in range(5):
        train_epoch(model, device, trainloader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, device, testloader, criterion)
        scheduler.step(val_loss)

        # --- PRUNING: Oprim trial-urile proaste devreme ---
        trial.report(val_acc, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
        # --------------------------------------------------

    return val_acc

# -------------------------
# 6. Main Execution
# -------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 1. Încărcăm datele O SINGURĂ DATĂ
    print("Loading data...")
    trainloader, testloader = prepare_data(args.batch_size)

    # 2. Modul OPTUNA (dacă este selectat)
    if args.mode == "aimd_optuna":
        print("\n=== Starting Optuna Hyperparameter Search ===")
        # MedianPruner oprește trial-urile care sunt sub media celorlalte
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        
        # Rulăm 15 trial-uri (suficient pentru a găsi ceva bun rapid)
        study.optimize(lambda t: optuna_objective(t, args, trainloader, testloader), n_trials=15)

        print("\n------------------------------------------------")
        print("Best Params Found:", study.best_params)
        print("------------------------------------------------\n")

        # Setăm parametrii optimi pentru antrenarea finală
        args.lr_init = study.best_params["lr_init"]
        args.add_amount = study.best_params["add_amount"]
        args.mult_factor = study.best_params["mult_factor"]
        args.mode = "aimd"  # Trecem în modul AIMD normal pentru rularea finală

    # 3. Configurare Finală a Modelului
    print(f"Initializing Final Training with Mode: {args.mode}")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    if args.mode == "sgd_fix":
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=5e-4)
        scheduler = None
    elif args.mode == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=5e-4)
        scheduler = None
    elif args.mode == "aimd":
        print(f"AIMD Config -> Init LR: {args.lr_init:.5f}, Add: {args.add_amount:.5f}, Mult: {args.mult_factor:.2f}")
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=5e-4)
        scheduler = AIMDScheduler(optimizer, args.lr_init, args.add_amount, args.mult_factor, args.min_lr, args.max_lr)
    else:
        raise ValueError("Unknown mode")

    # 4. Bucla Principală de Antrenare (30 Epoci)
    history = []
    print("\n=== Starting Final Training Loop (30 Epochs) ===")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, device, trainloader, criterion, optimizer)
        te_loss, te_acc = eval_epoch(model, device, testloader, criterion)
        
        # Step scheduler
        lr = scheduler.step(te_loss) if scheduler else optimizer.param_groups[0]["lr"]

        history.append([epoch, tr_loss, tr_acc, te_loss, te_acc, lr])
        print(f"Epoch {epoch}/{args.epochs} | Train Acc: {tr_acc:.3f} | Test Acc: {te_acc:.3f} | LR: {lr:.6f} | Time: {time.time()-t0:.1f}s")

    # 5. Salvare Rezultate
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(history, columns=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"])
    
    csv_path = f"{args.out_dir}/history_{args.mode}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nHistory saved to {csv_path}")

    # Plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(df.epoch, df.train_loss, label="train")
    plt.plot(df.epoch, df.test_loss, label="test")
    plt.title("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(df.epoch, df.train_acc, label="train")
    plt.plot(df.epoch, df.test_acc, label="test")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(df.epoch, df.lr, label="LR", color='green')
    plt.title("Learning Rate (AIMD Sawtooth)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{args.out_dir}/plots_{args.mode}.png"
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="aimd_optuna", choices=["sgd_fix", "adam", "aimd", "aimd_optuna"])
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for final training")
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Default params (used if mode != aimd_optuna)
    parser.add_argument("--lr_init", type=float, default=0.01)
    parser.add_argument("--add_amount", type=float, default=1e-4)
    parser.add_argument("--mult_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=0.5)
    
    parser.add_argument("--out_dir", type=str, default="results")
    
    args = parser.parse_args()
    main(args)