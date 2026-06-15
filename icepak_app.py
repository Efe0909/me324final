"""
Icepak Manifold Optimization Agent — Streamlit UI
Four tabs: Configure → Generate → Upload Results → Explore
All state persists in icepak_manifold.db.

Run:
    streamlit run icepak_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from pathlib import Path
import icepak_db as db
from icepak_agent import IcepakManifoldAgent, DEFAULT_DOFS

st.set_page_config(page_title="Icepak Agent", layout="wide")
st.title("Icepak Manifold Agent")

agent = IcepakManifoldAgent()

_RBF_SCALE = np.array([9.0, 0.030, 0.004, 0.013])

def _fit_rbf(pts):
    X = np.array([[p['dof_N'], p['dof_w'], p['dof_tb'], p['dof_H']] for p in pts])
    y = np.array([p['performance_score'] for p in pts], dtype=float)
    rbf = RBFInterpolator(X / _RBF_SCALE, y, kernel='thin_plate_spline', smoothing=0.5)
    return rbf, X, y

tab_cfg, tab_gen, tab_up, tab_explore = st.tabs([
    "⚙️ Configure", "🔄 Generate Batch", "📥 Upload Results", "🗺️ Explore"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Configure
# ══════════════════════════════════════════════════════════════════════
with tab_cfg:
    meta = db.get_metadata()
    dofs = db.get_dof_configs()

    _cfg_col, _info_col = st.columns([1, 2])

    # ── Left: inputs ───────────────────────────────────────────────────
    with _cfg_col:
        st.subheader("Run parameters")

        batch_size   = st.number_input("Batch size",            4,    32,   int(meta['batch_size'])                if meta else 16)
        alpha        = st.number_input("Alpha (decay)",         0.5,  1.0,  float(meta['alpha'])                  if meta else 0.85, step=0.05, format="%.2f")
        init_explore = st.number_input("Initial explore ratio", 0.05, 1.0,  float(meta['current_explore_ratio'])  if meta else 0.30, step=0.05, format="%.2f")

        st.subheader("DOF bounds")
        st.caption("min / max / step — step used for quantisation")

        dof_defaults = {d['param_name']: d for d in dofs} if dofs else \
                       {d['param_name']: d for d in DEFAULT_DOFS}

        new_dofs = []
        for d in DEFAULT_DOFS:
            name = d['param_name']
            src  = dof_defaults.get(name, d)
            st.markdown(f"**{name}**")
            ca, cb, cc = st.columns(3)
            mn  = ca.number_input("min",  value=float(src['min_val']),   key=f"mn_{name}", format="%.4f", label_visibility="collapsed")
            mx  = cb.number_input("max",  value=float(src['max_val']),   key=f"mx_{name}", format="%.4f", label_visibility="collapsed")
            stp = cc.number_input("step", value=float(src['step_size']), key=f"st_{name}", format="%.4f", label_visibility="collapsed")
            ca.caption("min"); cb.caption("max"); cc.caption("step")
            new_dofs.append({'param_name': name, 'min_val': mn, 'max_val': mx, 'step_size': stp})

        st.markdown("---")
        if st.button("💾 Save configuration", use_container_width=True):
            db.upsert_metadata(
                batch_size=int(batch_size),
                alpha=float(alpha),
                current_explore_ratio=float(init_explore),
                previous_explore_ratio=float(init_explore),
                next_batch_number=meta['next_batch_number'] if meta else 1,
            )
            db.set_dof_configs(new_dofs)
            st.success("Saved.")
            st.rerun()

        if st.button("🗑️ Reset database", type="secondary", use_container_width=True):
            db.reset_db()
            st.warning("Database wiped.")
            st.rerun()

    # ── Right: info ────────────────────────────────────────────────────
    with _info_col:
        s = db.stats()
        m = db.get_metadata()

        st.subheader("Status")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Evaluated points", s['evaluated'])
        mc2.metric("Staged points",    s['staged'])
        mc3.metric("Best T so far",    f"{s['best_T']:.1f} °C" if s['best_T'] else "—")
        mc4.metric("Next batch #",     m['next_batch_number'] if m else "—")

        st.markdown("---")
        st.subheader("Sampling quality")
        st.caption("Updates live as you edit bounds — before saving.")

        _pts_eval = db.get_evaluated()
        if not _pts_eval:
            st.info("No evaluated points yet — seed or run the agent first.")
        else:
            _bnd = {d['param_name']: (d['min_val'], d['max_val'], d['step_size'])
                    for d in new_dofs}

            def _in_bounds(p):
                return (
                    _bnd['N'][0]  <= p['dof_N']  <= _bnd['N'][1]  and
                    _bnd['w'][0]  <= p['dof_w']  <= _bnd['w'][1]  and
                    _bnd['tb'][0] <= p['dof_tb'] <= _bnd['tb'][1] and
                    _bnd['H'][0]  <= p['dof_H']  <= _bnd['H'][1]
                )

            _pts_in = [p for p in _pts_eval if _in_bounds(p)]
            _n_in   = len(_pts_in)
            _n_all  = len(_pts_eval)

            def _n_cells(lo, hi, step):
                return max(1, round((hi - lo) / step) + 1)

            _total_cells = (
                _n_cells(*_bnd['N']) * _n_cells(*_bnd['w']) *
                _n_cells(*_bnd['tb']) * _n_cells(*_bnd['H'])
            )
            _coverage = min(100.0, _n_in / _total_cells * 100)

            sq1, sq2, sq3, sq4 = st.columns(4)
            sq1.metric(
                "Points in bounds",
                f"{_n_in} / {_n_all}",
                help="Evaluated points within current DOF bounds vs total in DB."
            )
            sq2.metric(
                "Grid coverage",
                f"{_coverage:.1f} %",
                help=f"{_n_in} sampled out of {_total_cells:,} possible discrete grid cells."
            )

            if _n_in >= 2:
                _X   = np.array([[p['dof_N'], p['dof_w'], p['dof_tb'], p['dof_H']]
                                  for p in _pts_in])
                _lo  = np.array([_bnd[k][0] for k in ('N', 'w', 'tb', 'H')])
                _hi  = np.array([_bnd[k][1] for k in ('N', 'w', 'tb', 'H')])
                _rng = np.where((_hi - _lo) > 0, _hi - _lo, 1.0)
                _Xn  = (_X - _lo) / _rng

                _D  = cdist(_Xn, _Xn)
                np.fill_diagonal(_D, np.inf)
                _nn = _D.min(axis=1)

                _sparsity = float(_nn.mean()) / np.sqrt(len(_bnd))
                _cv  = float(_nn.std()) / float(_nn.mean()) if _nn.mean() > 0 else 0.0
                _uni = 1.0 / (1.0 + _cv)

                sq3.metric(
                    "Sparsity index",
                    f"{_sparsity:.3f}",
                    help="Mean NN distance ÷ bounding-box diagonal (normalised [0,1]^4 space). "
                         "Higher = sparser = more room to explore."
                )
                sq4.metric(
                    "NN uniformity",
                    f"{_uni:.2f}",
                    help="1 / (1 + CV of NN distances). "
                         "1.0 = perfectly even; lower = clusters + voids."
                )

                with st.expander("NN distance distribution", expanded=False):
                    fig_sq, ax_sq = plt.subplots(figsize=(6, 2.2))
                    ax_sq.hist(_nn, bins=min(20, _n_in // 2 + 1),
                               color="#4C72B0", edgecolor="white", linewidth=0.4)
                    ax_sq.axvline(_nn.mean(), color="tomato", linewidth=1.5,
                                  label=f"mean {_nn.mean():.3f}")
                    ax_sq.set_xlabel("Nearest-neighbour distance (normalised)")
                    ax_sq.set_ylabel("Count")
                    ax_sq.legend(fontsize=8)
                    fig_sq.tight_layout()
                    st.pyplot(fig_sq, use_container_width=True)
                    plt.close(fig_sq)

                    _labels = ['N', 'w', 'tb', 'H']
                    _pct    = [float((_X[:, i].max() - _X[:, i].min()) / _rng[i] * 100)
                               for i in range(4)]
                    st.caption("Sampled range as % of bound span per DOF:")
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    for _col, _lbl, _p in zip([pc1, pc2, pc3, pc4], _labels, _pct):
                        _col.metric(_lbl, f"{_p:.0f} %")
            else:
                sq3.metric("Sparsity index", "—")
                sq4.metric("NN uniformity",  "—")
                st.caption("Need ≥ 2 points in bounds for distance metrics.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Generate Batch
# ══════════════════════════════════════════════════════════════════════
with tab_gen:
    meta   = db.get_metadata()
    staged = db.get_staged()
    s      = db.stats()

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("Current state")
        if meta:
            st.metric("Batch number",     meta['next_batch_number'])
            st.metric("Explore ratio",    f"{meta['current_explore_ratio']:.0%}")
            st.metric("Evaluated points", s['evaluated'])
            st.metric("GP fitted",        "Yes" if agent._gp_fitted else f"No (need ≥5, have {s['evaluated']})")
        else:
            st.warning("Not configured. Go to Configure tab first.")

        st.markdown("---")

        if staged:
            st.info(f"{len(staged)} points STAGED — download or discard before generating a new batch.")
            st.markdown(f"**Batch #{staged[0]['batch_number']}**")

            if meta:
                batch_arr = np.array([[p['dof_N'], p['dof_w'], p['dof_tb'], p['dof_H']]
                                       for p in staged])
                bnum     = staged[0]['batch_number']
                csv_path = agent.export_icepak_csv(batch_arr, bnum)
                st.download_button(
                    "⬇️ Download Icepak CSV",
                    csv_path.read_text(),
                    file_name=csv_path.name,
                    mime="text/plain",
                    use_container_width=True,
                )

            if st.button("🗑️ Discard staged batch", type="secondary", use_container_width=True):
                agent.discard_last_batch()
                st.warning("Staged batch discarded. Explore ratio rolled back.")
                st.rerun()

        else:
            if not meta:
                st.button("🔄 Generate next batch", disabled=True, use_container_width=True)
            elif st.button("🔄 Generate next batch", type="primary", use_container_width=True):
                with st.spinner("Running GP + LCB sampling…"):
                    batch, bnum = agent.generate_batch()
                st.success(f"Batch {bnum} generated — {len(batch)} points staged.")
                st.rerun()

        st.markdown("---")
        st.subheader("Batch registry")
        batches = db.get_batches()
        if batches:
            df_b = pd.DataFrame(batches)[['batch_number', 'source', 'status', 'n_points',
                                          'explore_ratio', 'created_at']]
            df_b['explore_ratio'] = df_b['explore_ratio'].apply(
                lambda x: f"{x:.0%}" if x is not None else "—"
            )
            df_b.columns = ['#', 'source', 'status', 'n', 'explore', 'created']
            st.dataframe(df_b.set_index('#'), use_container_width=True)
        else:
            st.caption("No batches yet.")

    with col_r:
        if staged:
            st.subheader(f"Staged batch #{staged[0]['batch_number']}")
            batch_arr = np.array([[p['dof_N'], p['dof_w'], p['dof_tb'], p['dof_H']]
                                   for p in staged])
            T_mu, T_std = agent.predict_batch(batch_arr)

            rows = []
            for i, (p, row) in enumerate(zip(staged, batch_arr), 1):
                r = {
                    "#":       i,
                    "N":       int(row[0]),
                    "w [cm]":  round(row[1] * 100, 1),
                    "tb [mm]": round(row[2] * 1000, 1),
                    "H [mm]":  round(row[3] * 1000, 1),
                }
                if T_mu is not None:
                    r["T̂ [°C]"] = round(float(T_mu[i-1]), 1)
                    r["σ̂ [°C]"] = round(float(T_std[i-1]), 1)
                    r["LCB"]    = round(float(T_mu[i-1]) - 1.96 * float(T_std[i-1]), 1)
                rows.append(r)

            st.dataframe(pd.DataFrame(rows).set_index("#"), use_container_width=True, height=560)

            if T_mu is not None:
                fig, ax = plt.subplots(figsize=(7, 2.5))
                ax.bar(range(1, len(T_mu)+1), T_mu, yerr=1.96*T_std,
                       color="steelblue", alpha=0.7, capsize=3)
                ax.axhline(80, color="tomato", ls="--", lw=1.2, label="80°C")
                ax.set_xlabel("Point #"); ax.set_ylabel("T̂ [°C]")
                ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)
        else:
            st.subheader("Custom batch")
            st.caption("Manually specify design points, then stage them like any other batch.")

            _inp_tab, _csv_tab = st.tabs(["✏️ Table", "📄 CSV upload"])

            # ── shared helpers ─────────────────────────────────────────
            def _validate_custom(df_raw):
                """Return (array [N,4] in SI units, error_str | None)."""
                required = {'N', 'w', 'tb', 'H'}
                missing  = required - set(df_raw.columns)
                if missing:
                    return None, f"Missing columns: {missing}"
                df = df_raw[list(required)].dropna()
                if df.empty:
                    return None, "No rows with complete data."
                errs = []
                if not df['N'].apply(lambda x: float(x) == int(float(x))).all():
                    errs.append("N must be integer")
                if errs:
                    return None, "; ".join(errs)
                arr = np.array([
                    df['N'].astype(int).values,
                    df['w'].astype(float).values,
                    df['tb'].astype(float).values,
                    df['H'].astype(float).values,
                ], dtype=float).T          # shape (n, 4)  — units match input
                return arr, None

            def _stage_custom(arr_si):
                """arr_si: (n,4) in SI — N integer, w/tb/H in metres."""
                bnum = meta['next_batch_number']
                db.insert_staged(bnum, arr_si, explore_ratio=None, source='custom')
                db.upsert_metadata(next_batch_number=bnum + 1)

            # ── Tab A: editable table ──────────────────────────────────
            with _inp_tab:
                st.caption("Units: N = count, w = cm, tb = mm, H = mm")
                _n_rows = st.number_input("Start with N rows", 1, 64,
                                          meta['batch_size'] if meta else 8,
                                          key="cust_nrows")

                _default = pd.DataFrame({
                    'N':      [6]   * int(_n_rows),
                    'w':      [7.0] * int(_n_rows),
                    'tb':     [5.0] * int(_n_rows),
                    'H':      [35.0]* int(_n_rows),
                })
                _edited = st.data_editor(
                    _default,
                    column_config={
                        'N':  st.column_config.NumberColumn('N',      min_value=1,   max_value=20,  step=1,   format='%d'),
                        'w':  st.column_config.NumberColumn('w [cm]', min_value=0.1, max_value=30,  step=0.5, format='%.1f'),
                        'tb': st.column_config.NumberColumn('tb [mm]',min_value=0.1, max_value=50,  step=0.5, format='%.1f'),
                        'H':  st.column_config.NumberColumn('H [mm]', min_value=1.0, max_value=200, step=1.0, format='%.1f'),
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key="cust_table",
                )

                if st.button("Stage table batch", type="primary",
                             use_container_width=True, key="stage_tbl"):
                    _arr, _err = _validate_custom(_edited)
                    if _err:
                        st.error(_err)
                    else:
                        # convert display units → SI
                        _arr_si = _arr.copy()
                        _arr_si[:, 1] /= 100    # cm → m
                        _arr_si[:, 2] /= 1000   # mm → m
                        _arr_si[:, 3] /= 1000   # mm → m
                        _stage_custom(_arr_si)
                        st.success(f"{len(_arr_si)} points staged as batch "
                                   f"#{meta['next_batch_number']}.")
                        st.rerun()

            # ── Tab B: CSV upload ──────────────────────────────────────
            with _csv_tab:
                st.caption(
                    "CSV must have columns **N, w, tb, H**. "
                    "Units: N = count, w = cm, tb = mm, H = mm. "
                    "Extra columns are ignored."
                )
                _up = st.file_uploader("Drop CSV here", type=["csv"],
                                       key="cust_csv_upload")
                if _up:
                    import io as _io
                    _df_csv = pd.read_csv(_io.BytesIO(_up.read()))
                    _df_csv.columns = _df_csv.columns.str.strip()

                    # Auto-detect: if any column looks like it has SI headers
                    # (e.g. "w [cm]" → rename to "w")
                    _rename = {}
                    for col in _df_csv.columns:
                        _clean = col.split('[')[0].strip().lower()
                        if _clean in ('n', 'w', 'tb', 'h'):
                            _rename[col] = _clean.upper()
                    if _rename:
                        _df_csv = _df_csv.rename(columns=_rename)

                    _arr, _err = _validate_custom(_df_csv)
                    if _err:
                        st.error(_err)
                    else:
                        # Show preview with SI-converted values
                        _prev = pd.DataFrame({
                            'N':      _arr[:, 0].astype(int),
                            'w [cm]': _arr[:, 1].round(2),
                            'tb [mm]':_arr[:, 2].round(2),
                            'H [mm]': _arr[:, 3].round(1),
                        })
                        st.dataframe(_prev, use_container_width=True, height=260)
                        st.caption(f"{len(_arr)} rows parsed.")

                        if st.button("Stage CSV batch", type="primary",
                                     use_container_width=True, key="stage_csv"):
                            _arr_si = _arr.copy()
                            _arr_si[:, 1] /= 100
                            _arr_si[:, 2] /= 1000
                            _arr_si[:, 3] /= 1000
                            _stage_custom(_arr_si)
                            st.success(f"{len(_arr_si)} points staged as batch "
                                       f"#{meta['next_batch_number']}.")
                            st.rerun()


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Upload Results
# ══════════════════════════════════════════════════════════════════════
with tab_up:
    st.subheader("Upload Icepak iteration CSV")
    staged = db.get_staged()

    if not staged:
        st.warning("No staged points. Generate a batch first (Tab 2).")
    else:
        st.caption(f"{len(staged)} staged points waiting for results  "
                   f"(batch #{staged[0]['batch_number']})")

        uploaded = st.file_uploader(
            "Drag the iteration*.csv from Icepak here",
            type=["csv"],
            key="result_upload",
        )

        if uploaded:
            import io
            text   = uploaded.read().decode()
            df_raw = pd.read_csv(io.StringIO(text))
            df_raw.columns = df_raw.columns.str.strip('"').str.strip()
            df_raw['Trial'] = df_raw['Trial'].astype(str).str.strip('"').str.strip()
            df_raw = df_raw[df_raw['Trial'].str.startswith('trial')]

            preview_rows = []
            for i, p in enumerate(staged):
                trial_name = f"trial{i+1:03d}"
                match = df_raw[df_raw['Trial'] == trial_name]
                preview_rows.append({
                    "Trial":      trial_name,
                    "N":          int(p['dof_N']),
                    "w [cm]":     round(p['dof_w']  * 100, 1),
                    "tb [mm]":    round(p['dof_tb'] * 1000, 1),
                    "H [mm]":     round(p['dof_H']  * 1000, 1),
                    "Tmax [°C]": float(match['Tmax'].iloc[0]) if not match.empty else None,
                    "matched":    not match.empty,
                })

            df_preview = pd.DataFrame(preview_rows)
            n_matched  = df_preview['matched'].sum()
            good       = df_preview[df_preview['matched']]

            st.dataframe(df_preview[['Trial','N','w [cm]','tb [mm]','H [mm]','Tmax [°C]']],
                         use_container_width=True, height=420)

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Matched",   n_matched)
            cc2.metric("Unmatched", len(staged) - n_matched)
            if not good.empty:
                cc3.metric("Best in upload", f"{good['Tmax [°C]'].min():.1f} °C")

            st.markdown("---")
            if n_matched == 0:
                st.error("No trials matched. Check that the file is for the current staged batch.")
            elif st.button(f"✅ Commit {n_matched} results to DB",
                           type="primary", use_container_width=True):
                n, msg = agent.commit_results(text)
                st.success(msg)
                st.rerun()

    st.markdown("---")
    st.subheader("Evaluated history")

    # Auto-recompute manifold predictions every time this tab renders
    _pts_enabled = db.get_evaluated()
    if len(_pts_enabled) >= 5:
        _rbf_hist, _, _ = _fit_rbf(_pts_enabled)
        _pts_all_hist   = db.get_all_evaluated()
        _X_all_hist = np.array([[p['dof_N'], p['dof_w'], p['dof_tb'], p['dof_H']]
                                 for p in _pts_all_hist])
        _preds_hist = _rbf_hist(_X_all_hist / _RBF_SCALE)
        db.set_manifold_preds([(p['point_id'], float(v))
                               for p, v in zip(_pts_all_hist, _preds_hist)])

    pts_all = db.get_all_evaluated()
    if pts_all:
        df_hist = pd.DataFrame(pts_all).rename(columns={
            'dof_N': 'N', 'dof_w': 'w', 'dof_tb': 'tb',
            'dof_H': 'H', 'performance_score': 'T_chip',
            'manifold_pred': 'T_pred',
        })
        df_hist['w']        *= 100
        df_hist['tb']       *= 1000
        df_hist['H']        *= 1000
        df_hist['N']         = df_hist['N'].astype(int)
        df_hist['disabled']  = df_hist['disabled'].astype(bool)
        df_hist['error']     = df_hist['T_pred'] - df_hist['T_chip']
        df_hist = df_hist.sort_values('T_chip').reset_index(drop=True)

        orig_disabled = {p['point_id']: bool(p['disabled']) for p in pts_all}

        df_show  = df_hist.rename(columns={'w': 'w [cm]', 'tb': 'tb [mm]', 'H': 'H [mm]'})
        df_show  = df_show[['point_id', 'batch_number', 'N', 'w [cm]', 'tb [mm]',
                             'H [mm]', 'T_chip', 'T_pred', 'error', 'disabled']]
        _style = (df_show.style
                  .background_gradient(subset=['T_chip'], cmap='RdYlGn_r')
                  .background_gradient(subset=['error'],  cmap='RdBu_r', vmin=-15, vmax=15))

        edited = st.data_editor(
            _style,
            column_config={
                'point_id':     st.column_config.NumberColumn('id',          disabled=True),
                'batch_number': st.column_config.NumberColumn('batch',       disabled=True),
                'N':            st.column_config.NumberColumn('N',           disabled=True),
                'w [cm]':       st.column_config.NumberColumn('w [cm]',      disabled=True, format='%.1f'),
                'tb [mm]':      st.column_config.NumberColumn('tb [mm]',     disabled=True, format='%.1f'),
                'H [mm]':       st.column_config.NumberColumn('H [mm]',      disabled=True, format='%.1f'),
                'T_chip':       st.column_config.NumberColumn('T sim [°C]',  disabled=True, format='%.1f'),
                'T_pred':       st.column_config.NumberColumn('T RBF [°C]',  disabled=True, format='%.1f'),
                'error':        st.column_config.NumberColumn('error [°C]',  disabled=True, format='%+.1f',
                                    help='T_RBF − T_sim. Positive = surrogate predicts hotter than reality.'),
                'disabled':     st.column_config.CheckboxColumn('disabled',
                                    help='Exclude from GP and surrogate'),
            },
            use_container_width=True,
            height=400,
            hide_index=True,
            key="hist_editor",
        )

        # Auto-save on any checkbox change — no button needed
        changes = [
            (int(row['point_id']), int(row['disabled']))
            for _, row in edited.iterrows()
            if bool(row['disabled']) != orig_disabled.get(int(row['point_id']), False)
        ]
        if changes:
            db.set_points_disabled(changes)
            st.rerun()
    else:
        st.info("No evaluated points yet.")


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — 4D Surrogate Slice Explorer
# ══════════════════════════════════════════════════════════════════════
with tab_explore:

    # ── Load data from DB ──────────────────────────────────────────────
    _pts = db.get_evaluated()

    if len(_pts) < 5:
        st.warning("Not enough evaluated points to build surrogate (need ≥ 5). Seed the DB first.")
    else:
        _X_raw = np.array([[p['dof_N'], p['dof_w'], p['dof_tb'], p['dof_H']] for p in _pts])
        _y_raw = np.array([p['performance_score'] for p in _pts], dtype=float)

        _finite = _y_raw[np.isfinite(_y_raw)]
        T_db_min = int(np.floor(_finite.min())) if len(_finite) else 54
        T_db_max = int(np.ceil(_finite.max()))  if len(_finite) else 128

        # RBF — always refit so disabled-point changes are reflected immediately
        _SCALE = _RBF_SCALE
        _rbf   = RBFInterpolator(_X_raw / _SCALE, _y_raw,
                                 kernel="thin_plate_spline", smoothing=0.5)

        # Trust score per evaluated point: 1.0 = surrogate nails it, 0 = ≥15 °C off
        _preds_pts = _rbf(_X_raw / _SCALE)
        _errors_pts = _preds_pts - _y_raw          # positive = surrogate overestimates
        _trust_pts  = np.clip(1.0 - np.abs(_errors_pts) / 15.0, 0.0, 1.0)

        # ── Dynamic ranges from DB ─────────────────────────────────────
        N_vals  = sorted(int(v) for v in np.unique(_X_raw[:, 0]))
        tb_vals = sorted(round(v * 1000, 1) for v in np.unique(_X_raw[:, 2]))

        _DIMS = {
            'N':  dict(idx=0, lo=N_vals[0],   hi=N_vals[-1],   step=1,   scale=1,    unit='',   fmt='%.0f', integer=True,  discrete=N_vals),
            'w':  dict(idx=1, lo=5.0,          hi=9.5,          step=0.5, scale=100,  unit='cm', fmt='%.1f', integer=False, discrete=None),
            'tb': dict(idx=2, lo=tb_vals[0],   hi=tb_vals[-1],  step=0.5, scale=1000, unit='mm', fmt='%.1f', integer=False, discrete=tb_vals),
            'H':  dict(idx=3, lo=25.0,         hi=43.0,         step=1.0, scale=1000, unit='mm', fmt='%.0f', integer=False, discrete=None),
        }
        _KEYS = list(_DIMS.keys())

        def _dim_label(k):
            u = _DIMS[k]['unit']
            return f"{k} [{u}]" if u else k

        def _snap(k, v):
            d = _DIMS[k]
            if d['discrete']:
                return min(d['discrete'], key=lambda x: abs(x - v))
            v = max(d['lo'], min(d['hi'], v))
            v = round(v / d['step']) * d['step']
            return int(v) if d['integer'] else round(v, 6)

        # ── Session state ──────────────────────────────────────────────
        st.session_state.setdefault('sv_N',  N_vals[len(N_vals) // 2])
        st.session_state.setdefault('sv_w',  7.5)
        st.session_state.setdefault('sv_tb', tb_vals[len(tb_vals) // 2])
        st.session_state.setdefault('sv_H',  36.0)
        st.session_state.setdefault('exp_sel', -1)

        def _jump_to(i):
            row = _X_raw[i]
            st.session_state['sv_N']    = int(row[0])
            st.session_state['sv_w']    = _snap('w',  row[1] * 100)
            st.session_state['sv_tb']   = _snap('tb', row[2] * 1000)
            st.session_state['sv_H']    = _snap('H',  row[3] * 1000)
            st.session_state['exp_sel'] = i

        # ── Layout ────────────────────────────────────────────────────
        ctrl, plot_col, rank_col = st.columns([1, 2.5, 1.2])

        with ctrl:
            st.subheader("Axes")
            x_key  = st.selectbox("X axis", _KEYS, index=1)
            y_opts = [k for k in _KEYS if k != x_key]
            y_key  = st.selectbox("Y axis", y_opts,
                                  index=y_opts.index('H') if 'H' in y_opts else 2)

            slider_keys = [k for k in _KEYS if k not in (x_key, y_key)]

            st.subheader("Fixed values")
            fixed = {}
            for k in slider_keys:
                d = _DIMS[k]
                if d['discrete']:
                    fixed[k] = st.select_slider(
                        _dim_label(k), options=d['discrete'], key=f"sv_{k}")
                elif d['integer']:
                    fixed[k] = st.slider(
                        _dim_label(k), int(d['lo']), int(d['hi']),
                        step=1, key=f"sv_{k}")
                else:
                    fixed[k] = st.slider(
                        _dim_label(k), d['lo'], d['hi'],
                        step=d['step'], format=d['fmt'], key=f"sv_{k}")

            st.markdown("---")
            T_target = st.slider("Target T [°C]", 55, 125, 80)
            n_grid   = st.select_slider("Grid density", [50, 80, 120, 180], value=80)

        # ── Evaluate surrogate on 2D slice ────────────────────────────
        dx, dy = _DIMS[x_key], _DIMS[y_key]

        # Extent = actual data coverage on these axes + 5% extrapolation each side
        x_data = _X_raw[:, dx['idx']] * dx['scale']
        y_data = _X_raw[:, dy['idx']] * dy['scale']
        x_pad  = (x_data.max() - x_data.min()) * 0.05
        y_pad  = (y_data.max() - y_data.min()) * 0.05
        x_lo, x_hi = x_data.min() - x_pad, x_data.max() + x_pad
        y_lo, y_hi = y_data.min() - y_pad, y_data.max() + y_pad

        x_lin = np.linspace(x_lo, x_hi, n_grid)
        y_lin = np.linspace(y_lo, y_hi, n_grid)
        XX_d, YY_d = np.meshgrid(x_lin, y_lin)

        grid_pts = np.zeros((XX_d.size, 4))
        grid_pts[:, dx['idx']] = XX_d.ravel() / dx['scale']
        grid_pts[:, dy['idx']] = YY_d.ravel() / dy['scale']
        for k, v in fixed.items():
            grid_pts[:, _DIMS[k]['idx']] = v / _DIMS[k]['scale']

        T_grid = _rbf(grid_pts / _SCALE).reshape(XX_d.shape)

        # ── Classify data points ───────────────────────────────────────
        sel_idx = st.session_state['exp_sel']
        gray_x, gray_y                          = [], []
        slice_x, slice_y, slice_t, slice_trust  = [], [], [], []
        sel_pt                                  = None

        for i, (row, T_pt, trust) in enumerate(zip(_X_raw, _y_raw, _trust_pts)):
            px = row[dx['idx']] * dx['scale']
            py = row[dy['idx']] * dy['scale']
            in_slice = all(
                abs(row[_DIMS[k]['idx']] - fixed[k] / _DIMS[k]['scale'])
                <= _DIMS[k]['step'] / _DIMS[k]['scale'] * 0.55
                for k in slider_keys
            )
            if i == sel_idx:
                sel_pt = (px, py, T_pt, trust)
            elif in_slice:
                slice_x.append(px); slice_y.append(py)
                slice_t.append(T_pt); slice_trust.append(trust)
            else:
                gray_x.append(px); gray_y.append(py)

        n_in_slice = len(slice_x)

        # ── Matplotlib figure ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))

        cf = ax.contourf(XX_d, YY_d, T_grid,
                         levels=np.linspace(T_db_min, T_db_max, 40),
                         cmap="RdYlGn_r", alpha=0.85)
        fig.colorbar(cf, ax=ax, label="T_chip [°C]", shrink=0.9)

        for T_ref, col, lw in [(70, "lime", 1.2), (80, "tomato", 1.2),
                                (T_target, "dodgerblue", 2.5)]:
            if T_db_min <= T_ref <= T_db_max:
                cs = ax.contour(XX_d, YY_d, T_grid, levels=[T_ref],
                                colors=[col], linewidths=lw, zorder=5)
                ax.clabel(cs, fmt=f" {T_ref}°C ", fontsize=9, inline=True, colors=[col])

        # Data points
        if gray_x:
            ax.scatter(gray_x, gray_y, c="gray", alpha=0.2, s=20, zorder=3)

        # Slice points: fill = trust (green→red), edge = pass/fail
        _trust_cmap = plt.get_cmap("RdYlGn")
        for px, py, T_pt, tr in zip(slice_x, slice_y, slice_t, slice_trust):
            ec = "white" if T_pt <= T_target else "#ff6b6b"
            fc = _trust_cmap(tr)
            ax.scatter(px, py, c=[fc], edgecolors=ec, s=100, zorder=8, linewidths=2)
            ax.annotate(f"{T_pt:.0f}", (px, py),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=8, fontweight="bold", zorder=9)

        if sel_pt:
            px, py, T_pt, tr = sel_pt
            ax.scatter(px, py, c=[_trust_cmap(tr)], edgecolors="#00e5ff",
                       s=200, zorder=10, linewidths=2.5, marker="*")
            ax.annotate(f"★ {T_pt:.0f}°C", (px, py),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=9, fontweight="bold", color="#00e5ff", zorder=11)


        # Axis limits match the contour extent (data coverage + 5% extrapolation)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        fixed_str = "  |  ".join(
            f"{_dim_label(k)} = {v:.{0 if _DIMS[k]['integer'] else 1}f}"
            for k, v in fixed.items()
        )
        ax.set_title(f"Slice at:  {fixed_str}", fontsize=10, pad=8)
        ax.set_xlabel(_dim_label(x_key), fontsize=12)
        ax.set_ylabel(_dim_label(y_key), fontsize=12)
        ax.grid(True, alpha=0.18)
        plt.tight_layout()

        with plot_col:
            st.pyplot(fig)
            plt.close(fig)

            T_min   = T_grid.min()
            idx_min = np.unravel_index(T_grid.argmin(), T_grid.shape)
            frac_ok = (T_grid <= T_target).mean() * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Min T in slice", f"{T_min:.1f} °C",
                      f"{_dim_label(x_key)}={XX_d[idx_min]:.1f}  "
                      f"{_dim_label(y_key)}={YY_d[idx_min]:.1f}")
            c2.metric(f"Area ≤ {T_target}°C", f"{frac_ok:.0f}%")
            c3.metric("ANSYS pts in slice", n_in_slice)

        # ── Right: ranked points ───────────────────────────────────────
        with rank_col:
            st.subheader("Best runs")
            st.caption("↵ = snap sliders to plane")
            order    = np.argsort(_y_raw)
            show_all = st.checkbox("show all", value=False)
            n_show   = len(order) if show_all else min(20, len(order))

            h0, h1, h2, h3, h4 = st.columns([0.5, 1.1, 1.6, 0.9, 0.7])
            h0.markdown("<small>#</small>",               unsafe_allow_html=True)
            h1.markdown("<small>T °C</small>",            unsafe_allow_html=True)
            h2.markdown("<small>N · w · tb · H</small>",  unsafe_allow_html=True)
            h3.markdown("<small>trust</small>",            unsafe_allow_html=True)

            for rank, i in enumerate(order[:n_show], 1):
                row, T_pt, tr, err = _X_raw[i], _y_raw[i], _trust_pts[i], _errors_pts[i]
                dot    = "🟢" if T_pt <= 70 else ("🟡" if T_pt <= 80 else "🔴")
                is_sel = (i == sel_idx)
                c0, c1, c2, c3, c4 = st.columns([0.5, 1.1, 1.6, 0.9, 0.7])
                c0.markdown(f"<small>{'**→**' if is_sel else rank}</small>",
                            unsafe_allow_html=True)
                c1.markdown(f"{dot} **{T_pt:.1f}**")
                c2.markdown(
                    f"<small>{int(row[0])} · {row[1]*100:.1f} · "
                    f"{row[2]*1000:.0f} · {row[3]*1000:.0f}</small>",
                    unsafe_allow_html=True,
                )
                c3.markdown(
                    f"<small>{tr*100:.0f}%<br>{err:+.1f}°C</small>",
                    unsafe_allow_html=True,
                )
                c4.button("↵", key=f"exp_{i}", on_click=_jump_to, args=(i,),
                          type="primary" if is_sel else "secondary",
                          use_container_width=True)
