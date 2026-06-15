"""
IcepakManifoldAgent — GP-backed surrogate with explore/exploit alpha schedule.
State is fully persisted in SQLite via icepak_db; this class holds no mutable
in-memory state between Streamlit rerenders.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import icepak_db as db

DEFAULT_DOFS = [
    {'param_name': 'N',  'min_val': 3,     'max_val': 12,    'step_size': 1},
    {'param_name': 'w',  'min_val': 0.050, 'max_val': 0.090, 'step_size': 0.005},
    {'param_name': 'tb', 'min_val': 0.003, 'max_val': 0.008, 'step_size': 0.001},
    {'param_name': 'H',  'min_val': 0.025, 'max_val': 0.045, 'step_size': 0.001},
]


class IcepakManifoldAgent:

    def __init__(self):
        db.init_db()
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=3,
        )
        self._gp_fitted = False
        self._X_mean = self._X_std = None
        self._refit_gp()

    # ── GP internals ───────────────────────────────────────────────────
    def _refit_gp(self):
        pts = db.get_evaluated()
        if len(pts) < 5:
            return
        X = np.array([[p['dof_N'], p['dof_w'], p['dof_tb'], p['dof_H']] for p in pts])
        y = np.array([p['performance_score'] for p in pts])
        self._X_mean = X.mean(axis=0)
        self._X_std  = X.std(axis=0) + 1e-8
        self.gp.fit((X - self._X_mean) / self._X_std, y)
        self._gp_fitted = True

    def _predict(self, X):
        Xs = (X - self._X_mean) / self._X_std
        T_mean, T_std = self.gp.predict(Xs, return_std=True)
        return T_mean, T_std

    # ── Bounds helpers ─────────────────────────────────────────────────
    def _bounds(self):
        cfgs = db.get_dof_configs()
        low  = np.array([c['min_val']   for c in cfgs])
        high = np.array([c['max_val']   for c in cfgs])
        step = np.array([c['step_size'] for c in cfgs])
        return low, high, step

    @staticmethod
    def _quantize(X, low, step):
        return np.round((X - low) / step) * step + low

    @staticmethod
    def _lhs(n, low, high):
        d = len(low)
        pts = np.zeros((n, d))
        for j in range(d):
            idx = np.random.permutation(n)
            pts[:, j] = (idx + np.random.uniform(size=n)) / n
        return low + pts * (high - low)

    # ── Core API ───────────────────────────────────────────────────────
    def generate_batch(self):
        meta   = db.get_metadata()
        low, high, step = self._bounds()
        n      = meta['batch_size']
        ratio  = meta['current_explore_ratio']
        bnum   = meta['next_batch_number']

        n_explore = max(1, int(round(n * ratio)))
        n_exploit = n - n_explore

        explore = self._lhs(n_explore, low, high)

        if self._gp_fitted and n_exploit > 0:
            cands   = self._lhs(50_000, low, high)
            cands   = self._quantize(cands, low, step)
            T_mu, T_sig = self._predict(cands)
            lcb     = T_mu - 1.96 * T_sig
            idx     = np.argsort(lcb)[:n_exploit]
            exploit = cands[idx]
        else:
            exploit = self._lhs(n_exploit, low, high)

        batch = np.vstack([exploit, explore])
        batch = np.clip(self._quantize(batch, low, step), low, high)

        db.insert_staged(bnum, batch, explore_ratio=ratio)
        db.upsert_metadata(
            previous_explore_ratio=ratio,
            current_explore_ratio=max(ratio * meta['alpha'], 0.05),
            next_batch_number=bnum + 1,
        )
        self._refit_gp()
        return batch, bnum

    def discard_last_batch(self):
        meta = db.get_metadata()
        db.discard_staged()
        db.upsert_metadata(
            current_explore_ratio=meta['previous_explore_ratio'],
            next_batch_number=max(1, meta['next_batch_number'] - 1),
        )

    def commit_results(self, icepak_csv_text):
        """
        Parse an Icepak iteration*.csv and match rows to STAGED points by
        trial order (trial001 = 1st staged point, etc.).
        """
        import io
        staged = db.get_staged()
        if not staged:
            return 0, "No staged points to commit."

        df = pd.read_csv(io.StringIO(icepak_csv_text))
        df.columns = df.columns.str.strip('"').str.strip()
        df['Trial'] = df['Trial'].str.strip('"').str.strip()

        pairs = []
        for i, s in enumerate(staged):
            trial_name = f"trial{i+1:03d}"
            row = df[df['Trial'] == trial_name]
            if not row.empty:
                score = float(row['Tmax'].iloc[0])
                pairs.append((s['point_id'], score))

        db.commit_scores(pairs)
        self._refit_gp()
        return len(pairs), f"Committed {len(pairs)} / {len(staged)} points."

    def predict_batch(self, batch):
        if not self._gp_fitted:
            return None, None
        return self._predict(batch)

    def export_icepak_csv(self, batch, batch_number, base_path="/Volumes/storage"):
        out = Path(base_path) / f"batch{batch_number}" / "icepak_sparse.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        n = len(batch)
        tb_vals = sorted(set(f"{v:.3f}" for v in batch[:, 2]))
        tb_discrete = ",".join(tb_vals)

        lines = [
            "# TRIALS DATA", "#", "Trials,random", "#",
            "# VARIABLE INFORMATION",
            "# Variable Name,Discrete/In range,Values",
            f"H,,range,{batch[:,3].mean():.3f},{batch[:,3].min():.3f},{batch[:,3].max():.3f},0.001",
            f"N,,range,{int(batch[:,0].mean())},{int(batch[:,0].min())},{int(batch[:,0].max())},1",
            f"tb,,discrete,{tb_vals[0]},{tb_discrete}",
            f"w,,range,{batch[:,1].mean():.3f},{batch[:,1].min():.3f},{batch[:,1].max():.3f},0.005",
            "#",
            "# TRIALS INFORMATION",
            "# Trial Name,Trial selected,Restart ID,Restart type,Order",
        ]
        for i in range(n):
            lines.append(f"trial{i+1:03d},1,,0,{i+1}")
        lines += ["#", "# VARIABLE VALUES", "# Variable Names,H,N,tb,w,"]
        for i, r in enumerate(batch, 1):
            lines.append(f"trial{i:03d},{r[3]:.3f},{int(r[0])},{r[2]:.3f},{r[1]:.3f},")

        out.write_text("\n".join(lines) + "\n")
        return out
