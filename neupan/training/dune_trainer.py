"""
DUNE model trainer — generates data via convex optimization, then trains ObsPointNet.
"""

import os
import time
import pickle
import torch
import numpy as np
import cvxpy as cp
from colorama import deinit; deinit()
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim import Adam
from rich.console import Console
from rich.progress import Progress
from rich.live import Live

from neupan.utils.transforms import np_to_tensor, value_to_tensor, to_device


class PointDataset(Dataset):
    def __init__(self, inputs, labels, distances):
        self.inputs, self.labels, self.distances = inputs, labels, distances
    def __len__(self): return len(self.inputs)
    def __getitem__(self, i): return self.inputs[i], self.labels[i], self.distances[i]


class DUNETrain:
    """DUNE training pipeline.

    Args:
        model: ObsPointNet instance.
        robot_G, robot_h: Convex polygon inequality matrices.
        checkpoint_path: Directory for saving checkpoints.
    """

    def __init__(self, model, robot_G, robot_h, checkpoint_path):
        self.G, self.h, self.model = robot_G, robot_h, model
        self.checkpoint_path = checkpoint_path
        self._build_cvx_problem()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.console = Console()
        self.progress = Progress(transient=False)
        self.live = Live(self.progress, console=self.console, auto_refresh=False)
        self.loss_list = []

    # ==== public ====

    def start(self, data_size=100000, data_range=(-25, -25, 25, 25),
              batch_size=256, epoch=5000, valid_freq=100, save_freq=500,
              lr=5e-5, lr_decay=0.5, decay_freq=1500, save_loss=False, **kw):
        """Run full training. Returns path to last saved checkpoint."""
        self._save_config(locals())
        self.optimizer.param_groups[0]["lr"] = float(lr)

        print("generating dataset …")
        ds = self._gen_dataset(data_size, list(data_range))
        trn, val, _ = random_split(ds, [int(data_size * 0.8), int(data_size * 0.2), 0])
        tl, vl = DataLoader(trn, batch_size=batch_size), DataLoader(val, batch_size=batch_size)

        last_ckpt = None
        print("training …")
        with self.live:
            task = self.progress.add_task("[cyan]Training…", total=epoch)
            for i in range(epoch + 1):
                self.progress.update(task, advance=1); self.live.refresh()
                self.model.train(True)
                losses = self._run_epoch(tl, train=True)
                if i % valid_freq == 0:
                    self.model.eval()
                    vlosses = self._run_epoch(vl, train=False)
                    self._log(i, epoch, losses, vlosses, self.optimizer.param_groups[0]["lr"])
                if i % save_freq == 0:
                    last_ckpt = f"{self.checkpoint_path}/model_{i}.pth"
                    torch.save(self.model.state_dict(), last_ckpt)
                    print(f"saved model at epoch {i}")
                if (i + 1) % decay_freq == 0:
                    self.optimizer.param_groups[0]["lr"] *= lr_decay
                    print(f"lr → {self.optimizer.param_groups[0]['lr']}")
                total = sum(losses)
                self.loss_list.append(total)
                if save_loss:
                    with open(f"{self.checkpoint_path}/loss.pkl", "wb") as f:
                        pickle.dump(self.loss_list, f)

        print(f"done — model saved in {last_ckpt}")
        return last_ckpt

    def test(self, model_pth, train_dict_path, data_size_list, **kw):
        with open(train_dict_path, "rb") as f:
            td = pickle.load(f)
        model = to_device(td["model"])
        model.load_state_dict(torch.load(model_pth))
        ds = self._gen_dataset(max(data_size_list), td["data_range"])
        for n in data_size_list:
            dl = DataLoader(ds, batch_size=n)
            results = [self._test_batch(model, *batch, n) for batch in dl]
            avg = lambda idx: sum(r[0][idx] for r in results) / len(results)
            avg_t = sum(r[1] for r in results) / len(results)
            with open(os.path.dirname(model_pth) + "/test_results.txt", "a") as f:
                print(f"Model {os.path.basename(model_pth)} N={n} t={avg_t:.4f}s  "
                      f"mu={avg(0):.2e} dist={avg(1):.2e} fa={avg(2):.2e} fb={avg(3):.2e}", file=f)

    # ==== private ====

    def _build_cvx_problem(self):
        self._mu = cp.Variable((self.G.shape[0], 1), nonneg=True)
        self._p = cp.Parameter((2, 1))
        cost = self._mu.T @ (self.G.cpu() @ self._p - self.h.cpu())
        self._prob = cp.Problem(cp.Maximize(cost), [cp.norm(self.G.cpu().T @ self._mu) <= 1])

    def _solve(self, p):
        self._p.value = p
        self._prob.solve(solver=cp.ECOS)
        return self._prob.value, self._mu.value

    def _gen_dataset(self, n, data_range):
        inp, lab, dist = [], [], []
        pts = np.random.uniform(data_range[:2], data_range[2:], (n, 2))
        for p in pts:
            d, mu = self._solve(p.reshape(2, 1))
            inp.append(np_to_tensor(p.reshape(2, 1)))
            lab.append(np_to_tensor(mu))
            dist.append(value_to_tensor(d))
        return PointDataset(inp, lab, dist)

    def _run_epoch(self, loader, train=True):
        mu_l, dist_l, fa_l, fb_l = 0, 0, 0, 0
        for inp, lab_mu, lab_d in loader:
            self.optimizer.zero_grad()
            inp = torch.squeeze(inp)
            out = torch.unsqueeze(self.model(inp), 2)
            d = self._calc_dist(out, inp)
            lm, ld = self.loss_fn(out, lab_mu), self.loss_fn(d, lab_d)
            lfa, lfb = self._calc_fab_loss(out, lab_mu, inp)
            loss = lm + ld + lfa + lfb
            if train: loss.backward(); self.optimizer.step()
            mu_l += lm.item(); dist_l += ld.item(); fa_l += lfa.item(); fb_l += lfb.item()
        n = len(loader)
        return mu_l/n, dist_l/n, fa_l/n, fb_l/n

    def _test_batch(self, model, inp, lab_mu, lab_d, n):
        inp = torch.squeeze(inp)
        t0 = time.time()
        out = torch.unsqueeze(model(inp), 2)
        dt = time.time() - t0
        d = self._calc_dist(out, inp)
        lfa, lfb = self._calc_fab_loss(out, lab_mu, inp)
        return [self.loss_fn(out, lab_mu).item(), self.loss_fn(d, lab_d).item(), lfa.item(), lfb.item()], dt

    def _calc_dist(self, mu, inp):
        inp = torch.unsqueeze(inp, 2)
        return torch.squeeze(torch.bmm(mu.transpose(1, 2), self.G @ inp - self.h))

    def _calc_fab_loss(self, out_mu, lab_mu, inp):
        ip = torch.unsqueeze(inp, 2)
        theta = np.random.uniform(0, 2 * np.pi)
        R = np_to_tensor(np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta),  np.cos(theta)]]))
        fa  = (-R @ self.G.T @ out_mu).transpose(1, 2)
        fa2 = (-R @ self.G.T @ lab_mu).transpose(1, 2)
        fb  = fa  @ ip + out_mu.transpose(1, 2) @ self.h
        fb2 = fa2 @ ip + lab_mu.transpose(1, 2) @ self.h
        return self.loss_fn(fa, fa2), self.loss_fn(fb, fb2)

    def _save_config(self, cfg):
        d = {k: cfg[k] for k in ("data_size", "data_range", "batch_size", "epoch",
             "valid_freq", "save_freq", "lr", "lr_decay", "decay_freq")}
        d.update(robot_G=self.G, robot_h=self.h, model=self.model)
        with open(f"{self.checkpoint_path}/train_dict.pkl", "wb") as f:
            pickle.dump(d, f)

    def _log(self, i, total, tr, va, lr):
        fmt = lambda v: f"{v:.2e}"
        msg = (f"Epoch {i}/{total}  lr={lr}\n"
               f"  mu: {fmt(tr[0])} | {fmt(va[0])}   dist: {fmt(tr[1])} | {fmt(va[1])}\n"
               f"  fa: {fmt(tr[2])} | {fmt(va[2])}   fb:   {fmt(tr[3])} | {fmt(va[3])}")
        print(msg)
        with open(f"{self.checkpoint_path}/results.txt", "a") as f:
            print(msg, file=f)
