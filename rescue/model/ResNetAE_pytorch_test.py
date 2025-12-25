from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .ResNet_dilated_1d import *
from tqdm import tqdm
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import os
import numpy as np
import scanpy as sc
import matplotlib
import pandas as pd
matplotlib.use('Agg')

def lins_ccc(x, y):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    y_var = np.var(y)
    covariance = np.mean((x - x_mean) * (y - y_mean))
    ccc = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)
    return ccc

class ResNet_pred(torch.nn.Module):
    def __init__(self,
                 n_centroids=14):
        super(ResNet_pred, self).__init__()
        self.pre_resnet = seresnet1d18_dilated(num_classes=n_centroids)
        # self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.MSELoss()

    def loss_function(self, x, label_list, k, device):
        z = self.pre_resnet(x)
        # criterion = nn.MSELoss()
        labels = torch.tensor(label_list)
        labels = labels.to(device)
        # loss = criterion(z, labels)
        loss = self.criterion(z, labels)
        ccc_value = lins_ccc(z, labels)
        return loss, ccc_value

    def fit_res(self, adata, dataloader, dataloader_test, batch_size, k,
                lr=0.002,
                weight_decay=5e-4,
                device='cpu',
                n_epoch=1501,
                verbose=True,
                outdir=None,
                ):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        iteration = 0
        epoch_min = 0
        epoch_max = 0
        count = 0
        ccc_value_max = 0
        min_loss = 1

        import csv, os
        from datetime import datetime
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        log_csv = os.path.join(outdir if outdir else ".", "training_log.csv")
        loss_train_hist, loss_val_hist, ccc_val_hist = [], [], []

        # Use two rich Progress objects: one for epochs (with time columns),
        # and one for iterations (no time columns) so iteration bar stays compact.

        iter_progress = Progress(
            TextColumn("[green]{task.description}"),
            BarColumn(bar_width=None),
            " ",
            TextColumn("{task.completed}/{task.total}"),
        )

        epoch_progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            " ",
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with iter_progress as p_iter, epoch_progress as p_epoch:
            iter_task = p_iter.add_task("Iterations", total=0)
            epoch_task = p_epoch.add_task("Epochs", total=n_epoch)

            for epoch in range(n_epoch):
                self.train()
                count = count + 1

                train_loss_sum = 0.0
                train_batches = 0

                # reset iteration task for this epoch
                p_iter.update(iter_task, total=len(dataloader), completed=0, description=f"Iterations (epoch {epoch+1})")

                for i, (x, labels) in enumerate(dataloader):
                    # adapt input dims: x: (batch_size, sequence_length) --> (batch_size, 1, sequence_length)
                    x = x.view(x.size(0), 1, -1)  # single-channel sequence
                    x = x.to(torch.float).to(device)
                    optimizer.zero_grad()
                    loss, ccc_value = self.loss_function(x, labels, k, device)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
                    optimizer.step()

                    train_loss_sum += float(loss.detach().cpu())
                    train_batches += 1

                    # update iteration progress and show loss in the task console
                    p_iter.update(iter_task, advance=1, refresh=True)
                    # show latest loss in the iteration console (kept minimal)
                    p_iter.console.print(f"[cyan]loss={loss:.3f}[/cyan]", end="\r")
                    iteration += 1

                epoch_train_loss = train_loss_sum / max(1, train_batches)

                # mark iteration bar as complete for this epoch
                p_iter.update(iter_task, completed=len(dataloader))

                # advance epoch bar so it reflects progress
                p_epoch.update(epoch_task, advance=1)

                # ------------------------validation----------------------
                with torch.no_grad():
                    self.eval()
                    z_val, labels_all = [], []
                    for x, labels in dataloader_test:
                        # print(labels)
                        x = x.view(x.size(0), 1, -1)  # 单通道序列
                        x = x.to(torch.float)
                        x = x.to(device)

                        # optimizer.zero_grad()
                        labels = labels.to(device)

                        z = self.pre_resnet(x)

                        z_val.append(z)
                        labels_all.append(labels)

                    # 合并所有 batch 的输出和标签
                    z_val = torch.cat(z_val, dim=0)  # 形状：[N, 输出维度]
                    labels = torch.cat(labels_all, dim=0)  # 形状：[N] 或 [N, 1]
                    loss = self.criterion(z_val, labels)
                    ccc_value = lins_ccc(z_val, labels)

                    # loss, ccc_value = self.loss_function(x, labels, k, device)

                    # Track history for plotting
                    loss_train_hist.append(epoch_train_loss)
                    loss_val_hist.append(float(loss.detach().cpu()))
                    ccc_val_hist.append(float(ccc_value))

                    # --- append one row to CSV each epoch ---
                    if outdir:
                        write_header = not os.path.exists(log_csv)
                        with open(log_csv, "a", newline="") as f:
                            w = csv.writer(f)
                            if write_header:
                                w.writerow(["timestamp", "epoch", "train_loss", "val_loss", "val_ccc"])
                            w.writerow([datetime.now().isoformat(timespec="seconds"),
                                        epoch, epoch_train_loss, float(loss), float(ccc_value)])

                        # --- (re)draw curves every epoch ---
                        try:
                            import matplotlib.pyplot as plt
                            epochs = range(1, len(loss_train_hist) + 1)
                            fig, ax1 = plt.subplots(figsize=(8, 5))

                            # Loss curves (left y-axis)
                            l1, = ax1.plot(epochs, loss_train_hist, label="Train Loss")
                            l2, = ax1.plot(epochs, loss_val_hist, label="Val Loss", linestyle="--")
                            ax1.set_xlabel("Epoch")
                            ax1.set_ylabel("Loss")
                            ax1.grid(False)

                            # CCC curve (right y-axis)
                            ax2 = ax1.twinx()
                            l3, = ax2.plot(epochs, ccc_val_hist, label="Val CCC",color='orange')
                            ax2.set_ylabel("CCC")

                            # unified legend
                            lines = [l1, l2, l3]
                            labels = [ln.get_label() for ln in lines]
                            ax1.legend(lines, labels, loc="best")

                            fig.tight_layout()
                            fig.savefig(os.path.join(outdir, "loss_ccc_joint.png"), dpi=200)
                            plt.close(fig)
                        except Exception as e:
                            print(f"[warn] joint plot failed: {e}")

                    if loss < min_loss:
                    # if ccc_value_max < ccc_value:
                        count = 0
                        # epoch_min = epoch
                        epoch_max = epoch
                        min_loss = loss
                        ccc_value_max = ccc_value
                        if outdir:
                            sc.settings.figdir = outdir
                            torch.save(self.state_dict(), os.path.join(outdir, 'model.pt'))  # save model
                            # also save predictions for the best model
                            try:
                                if outdir:
                                    output_pre = z_val.cpu().numpy()
                                    column_indices = adata.obs.columns[:k]
                                    row_indices = adata.obs.index
                                    output_df = pd.DataFrame(output_pre, index=row_indices, columns=column_indices)
                                    output_df.to_csv(os.path.join(outdir, 'model_outputs.txt'), sep='\t', index=True, header=True)
                            except Exception as e:
                                continue
                        print(f"\nsave at epoch:{epoch}")
                if count == 500 or min_loss == 0:
                    print(
                        f"\nearly stop........\nepoch_max:{epoch_max},min_loss: {min_loss:.4f},ccc_value_min: {ccc_value_max}")
                    break

    def forward(self, x):
        return self.pre_resnet(x)
