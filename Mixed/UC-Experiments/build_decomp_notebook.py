"""Run this once to generate decomp_3grids.ipynb"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}

cells = []

# 0 title
cells.append(md(
    "# DECOMP — Mutable Line Limits\n"
    "## Applied to IEEE 14-bus, IEEE 39-bus, IEEE 57-bus\n\n"
    "**Algorithm:** DECOMP (Column-and-Constraint Generation, Yue et al. 2019) adapted "
    "for b-parameter counterfactual explanations in UC.\n\n"
    "Objective: find minimum change to line flow limits b such that the 20% emissions-reduction "
    "foil becomes UC-optimal.\n"
))

# 1 sys check
cells.append(code("import sys\nprint(sys.executable)\n"))

# 2 imports header
cells.append(md("## 0 · Imports and helpers"))

# 3 imports
cells.append(code(
    "import os, importlib, warnings\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.gridspec as gridspec\n"
    "from dataclasses import replace\n"
    "\n"
    "import gurobipy as gp\n"
    "from gurobipy import GRB\n"
    "\n"
    'os.environ["GRB_LICENSE_FILE"] = r"C:\\Users\\tomas\\Desktop\\gurobi.lic"\n'
    "\n"
    "import uc_pipeline, uc_data_loader, uc_branch_sandwich_4b, uc_master_relax_4b, uc_decomp_4b\n"
    "importlib.reload(uc_pipeline)\n"
    "importlib.reload(uc_data_loader)\n"
    "importlib.reload(uc_branch_sandwich_4b)\n"
    "importlib.reload(uc_master_relax_4b)\n"
    "importlib.reload(uc_decomp_4b)\n"
    "\n"
    "from uc_pipeline import (\n"
    "    NetworkUCData, IndexMap,\n"
    "    build_index_map_network_uc,\n"
    "    build_cost_vector_network_uc,\n"
    "    default_initial_conditions,\n"
    "    solve_uc_with_cost_4b,\n"
    "    make_emissions_foil_4b,\n"
    "    set_objective_from_cvec,\n"
    ")\n"
    "from uc_data_loader import quick_setup\n"
    "from uc_decomp_4b import UCDecomp4b\n"
    "from uc_branch_sandwich_4b import UCBranchAndSandwichWCE_4b\n"
    "\n"
    "# Color palette for dispatch stacked-area plots (matches BranchSand_3grids.ipynb)\n"
    'TECH_COLORS = ["#73726c","#378ADD","#BA7517","#7F77DD","#1D9E75",\n'
    '               "#D85A30","#639922","#5DCAA5","#EF9F27","#E24B4A"]\n'
    "\n"
    "DATA_DIR = r"
    '"C:\\Users\\tomas\\Documents\\GitHub\\Counterfactual-Explanations-for-optimization-problems\\Mixed\\UC-Experiments\\Data"\n'
    "ALPHA    = 0.10   # 10% emissions reduction (matches bs_7grids.ipynb)\n"
    "UTIL_THR = 0.75   # congestion threshold\n"
    'warnings.filterwarnings("ignore")\n'
    'print("Imports OK")\n'
    'print(f"alpha={ALPHA:.0%}  UTIL_THR={UTIL_THR:.0%}")\n'
))

# 4 helpers header
cells.append(md("### Helpers (oracle, line-limit utilities)"))

# 5 helpers
cells.append(code(
    "def replace_line_limits(data, b_line):\n"
    "    b_line = np.asarray(b_line, dtype=float).reshape(-1)\n"
    "    new_lines = [replace(L, fmax=float(b_line[ell])) for ell, L in enumerate(data.lines)]\n"
    "    return replace(data, lines=new_lines)\n"
    "\n"
    "\n"
    "def get_congested_free_lines(sol0, b0, thr=0.75):\n"
    "    f_abs_max = np.max(np.abs(sol0['f']), axis=1)\n"
    "    util      = f_abs_max / np.maximum(b0, 1e-9)\n"
    "    idx_cong  = [ell for ell in range(len(b0)) if util[ell] >= thr]\n"
    "    if not idx_cong:\n"
    "        idx_cong = [int(np.argmax(util))]\n"
    "    return idx_cong, util\n"
    "\n"
    "\n"
    "def build_b_bounds(b0, b_free_idx, scale_up=1.2):\n"
    "    bL = b0.copy(); bU = b0.copy()\n"
    "    for ell in b_free_idx:\n"
    "        bU[ell] = scale_up * b0[ell]\n"
    "    return bL, bU\n"
    "\n"
    "\n"
    "def make_line_weights(DATA, b0, util=None, alpha=1.0, beta=1.0, gamma=0.5, eps_u=0.05):\n"
    "    b_susc  = np.array([float(L.b) for L in DATA.lines], dtype=float)\n"
    "    x_proxy = 1.0 / np.maximum(np.abs(b_susc), 1e-9)\n"
    "    w = (x_proxy / max(np.median(x_proxy), 1e-9))**alpha\n"
    "    w *= (b0 / max(np.median(b0), 1e-9))**beta\n"
    "    if util is not None:\n"
    "        w *= (1.0 / (eps_u + util))**gamma\n"
    "    w = w / np.mean(w)\n"
    "    return np.clip(w, 0.1, 10.0)\n"
    "\n"
    "\n"
    "class UCWeakWCEOracle:\n"
    "    def __init__(self, data, cvec, idx, window_size, per_bus_neutrality,\n"
    "                 u_init, p_init, on_t, off_t,\n"
    "                 foil_extra_constr_fn=None, output_flag=0, time_limit=None, cache_decimals=3):\n"
    "        self.data = data\n"
    "        self.cvec = np.asarray(cvec, float)\n"
    "        self.idx  = idx\n"
    "        self.window_size        = int(window_size)\n"
    "        self.per_bus_neutrality = bool(per_bus_neutrality)\n"
    "        self.u_init = u_init; self.p_init = p_init\n"
    "        self.on_t = on_t;     self.off_t  = off_t\n"
    "        self.foil_extra_constr_fn = foil_extra_constr_fn\n"
    "        self.output_flag    = int(output_flag)\n"
    "        self.time_limit     = time_limit\n"
    "        self.cache_decimals = int(cache_decimals)\n"
    "        self.cache_plain = {}\n"
    "        self.cache_foil  = {}\n"
    "\n"
    "    def _key(self, b):\n"
    "        return tuple(np.round(np.asarray(b, float), self.cache_decimals))\n"
    "\n"
    "    def _solve(self, b, extra_fn, cache):\n"
    "        key = self._key(b)\n"
    "        if key in cache:\n"
    "            return cache[key]\n"
    "        data_b = replace_line_limits(self.data, b)\n"
    "        _, sol, z = solve_uc_with_cost_4b(\n"
    "            data=data_b, idx=self.idx, cvec=self.cvec,\n"
    "            window_size=self.window_size,\n"
    "            per_bus_neutrality=self.per_bus_neutrality,\n"
    "            u_init=self.u_init, p_init=self.p_init,\n"
    "            on_time_init=self.on_t, off_time_init=self.off_t,\n"
    "            extra_constr_fn=extra_fn,\n"
    "            output_flag=self.output_flag, time_limit=self.time_limit,\n"
    "        )\n"
    "        out = (None, None, None) if sol is None else (float(sol['obj']), np.array(z, float), sol)\n"
    "        cache[key] = out\n"
    "        return out\n"
    "\n"
    "    def solve_plain(self, b): return self._solve(b, None, self.cache_plain)\n"
    "    def solve_foil(self, b):  return self._solve(b, self.foil_extra_constr_fn, self.cache_foil)\n"
    "\n"
    "\n"
    'print("Helpers defined.")\n'
))

# ── Plot helper (DECOMP-specific port of plot_bs_results) ────────────────
cells.append(md("### Plot helper (`plot_decomp_results`)"))
cells.append(code(
    "def plot_decomp_results(DATA, b0, b_hat, sol0, sol_plain_hat, sol_foil_hat,\n"
    '                         res, label="", foil_label="emissions −0%", method="DECOMP"):\n'
    '    """4-panel summary figure for a DECOMP run.\n'
    "    Ported from BranchSand_3grids.ipynb\'s plot_bs_results to consume the\n"
    "    DECOMP result dict (which uses the same `success`, `b_hat`, `F_opt`,\n"
    "    `master_LB`, `certified` fields).\n"
    '    """\n'
    "    if res is None or not res.get('success'):\n"
    "        print(f'[{label}] No successful result — nothing to plot.')\n"
    "        return\n"
    "    nG = len(DATA.gens); nR = len(DATA.rens); T = int(DATA.T); nL = len(DATA.lines)\n"
    "    hrs = np.arange(T)\n"
    "    fig = plt.figure(figsize=(18, 11))\n"
    "    fig.suptitle(f'{method} — {label}   [foil: {foil_label}]', fontsize=13, fontweight='bold')\n"
    "    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.28)\n"
    "    ax1, ax2, ax3, ax4 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])\n"
    "\n"
    "    def _dispatch_panel(ax, sol, title):\n"
    "        if sol is None:\n"
    "            ax.set_title(f'{title} (N/A)'); return\n"
    "        p_raw = sol['p']\n"
    "        if isinstance(p_raw, dict):\n"
    "            p_mat = np.array([[float(p_raw[g, t]) for t in range(T)] for g in range(nG)])\n"
    "        else:\n"
    "            p_mat = np.asarray(p_raw, dtype=float)\n"
    "            if p_mat.shape == (T, nG): p_mat = p_mat.T\n"
    "        sol = dict(sol, p=p_mat)\n"
    "        bottom = np.zeros(T)\n"
    "        for g in range(nG):\n"
    "            col = TECH_COLORS[g % len(TECH_COLORS)]\n"
    "            ax.fill_between(hrs, bottom, bottom + sol['p'][g], color=col, alpha=0.75, label=f'G{g}', step='mid')\n"
    "            bottom += sol['p'][g]\n"
    "        if nR > 0 and 'curt' in sol:\n"
    "            avail = np.vstack([r.avail for r in DATA.rens])\n"
    "            ren_inj = (avail - sol['curt']).sum(axis=0)\n"
    "            ax.fill_between(hrs, bottom, bottom + ren_inj, color='#1D9E75', alpha=0.5, label='Renewable', step='mid')\n"
    "            bottom += ren_inj\n"
    "        ax.plot(hrs, bottom, color='orange', lw=1.8, label='Total production', zorder=5)\n"
    "        ax.plot(hrs, DATA.demand.sum(axis=0), 'k--', lw=1.5, label='Demand')\n"
    "        ax.set_title(title, fontsize=10); ax.set_xlabel('Hour'); ax.set_ylabel('MW')\n"
    "        ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=4, loc='upper left')\n"
    "\n"
    "    _dispatch_panel(ax1, sol0,         'Factual dispatch  (b = b₀)')\n"
    "    _dispatch_panel(ax2, sol_foil_hat, 'Counterfactual dispatch  (b = b̂, foil active)')\n"
    "\n"
    "    curtF  = sol0['curt'].sum(axis=0) if nR>0 and 'curt' in sol0 else np.zeros(T)\n"
    "    curtCF = sol_foil_hat['curt'].sum(axis=0) if sol_foil_hat is not None and nR>0 and 'curt' in sol_foil_hat else np.zeros(T)\n"
    "    curtP  = sol_plain_hat['curt'].sum(axis=0) if sol_plain_hat is not None and nR>0 and 'curt' in sol_plain_hat else np.zeros(T)\n"
    "    ax3.bar(hrs-0.27, curtF,  0.26, color='#D85A30', alpha=0.85, label=f'Factual  ({curtF.sum():.1f} MWh)')\n"
    "    ax3.bar(hrs,      curtP,  0.26, color='#BA7517', alpha=0.85, label=f'Plain@b̂  ({curtP.sum():.1f} MWh)')\n"
    "    ax3.bar(hrs+0.27, curtCF, 0.26, color='#378ADD', alpha=0.85, label=f'Foil@b̂   ({curtCF.sum():.1f} MWh)')\n"
    "    ax3.set_title('Curtailment per hour (MWh)', fontsize=10)\n"
    "    ax3.set_xlabel('Hour'); ax3.set_ylabel('MWh'); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis='y')\n"
    "\n"
    "    b_hat_pu = b_hat / np.maximum(b0, 1e-12)\n"
    "    changed  = np.where(np.abs(b_hat_pu - 1.0) > 1e-4)[0]\n"
    "    x_all    = np.arange(nL)\n"
    "    ax4.bar(x_all, b_hat_pu, color='#cccccc', alpha=0.6, label='Unchanged')\n"
    "    if len(changed):\n"
    "        ax4.bar(changed, b_hat_pu[changed], color='#378ADD', alpha=0.9, label='Changed')\n"
    "    ax4.axhline(1.0, color='k', lw=1.0, ls='--', label='Original (1.0 p.u.)')\n"
    "    ax4.set_title('Line limits — counterfactual vs original  (p.u.)', fontsize=10)\n"
    "    ax4.set_xlabel('Line index ℓ'); ax4.set_ylabel('b̂ / b₀  (p.u.)')\n"
    "    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis='y')\n"
    "    ax4.text(0.98, 0.97, f'{len(changed)} line(s) changed', transform=ax4.transAxes, ha='right', va='top', fontsize=8)\n"
    "    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()\n"
    "\n"
    "    F_best = float(res.get('F_opt', np.nan))\n"
    "    LB     = float(res.get('master_LB', np.nan))\n"
    "    gap_pct = (100.0 * max(0.0, F_best - LB) / abs(F_best)\n"
    "               if np.isfinite(F_best) and np.isfinite(LB) and abs(F_best) > 1e-12 else np.nan)\n"
    "    print(f'\\n{label} — {method} summary:')\n"
    "    print(f'  Iterations       : {res.get(\"iterations\")}')\n"
    "    print(f'  Patterns added   : {res.get(\"seen_patterns\")}')\n"
    "    print(f'  F_opt (incumbent): {F_best:.6f}')\n"
    "    print(f'  Master LB        : {LB:.6f}')\n"
    "    print(f'  Gap              : {gap_pct:.3f}%')\n"
    "    print(f'  Certified        : {res.get(\"certified\")}')\n"
    "    print(f'  Termination      : {res.get(\"termination_reason\")}')\n"
    "    cand = res.get('candidate')\n"
    "    if cand is not None and cand.get('F') is not None and cand['F'] < F_best - 1e-9:\n"
    "        print(f'  [candidate] master also found b with F={cand[\"F\"]:.4f} (oracle_ce_gap={cand.get(\"oracle_ce_gap\")}) — not certified')\n"
    "    if len(changed):\n"
    "        delta_abs = b_hat - b0\n"
    "        delta_pu  = delta_abs / np.maximum(b0, 1e-12)\n"
    "        util_arr  = np.max(np.abs(sol0['f']), axis=1) / np.maximum(b0, 1e-9)\n"
    "        print(f\"  {'ell':>4}  {'fr->to':>6}  {'b0(MW)':>9}  {'b_hat(MW)':>10}  {'Δ(MW)':>8}  {'b_hat(pu)':>10}  {'Δ(pu)':>8}  {'util':>6}\")\n"
    "        rows = sorted([(ell, int(DATA.lines[ell].fr), int(DATA.lines[ell].to),\n"
    "                        b0[ell], b_hat[ell], delta_abs[ell], b_hat_pu[ell], delta_pu[ell], util_arr[ell])\n"
    "                       for ell in changed], key=lambda r: -abs(r[7]))\n"
    "        for ell, fr, to, b0v, bhv, dv, bhpu, dpu, uti in rows:\n"
    "            print(f'  {ell:>4}  {fr:>2}->{to:<3}  {b0v:>9.3f}  {bhv:>10.3f}  {dv:>8.3f}  {bhpu:>10.3f}  {dpu:>+8.3f}  {uti:>6.3f}')\n"
    "    else:\n"
    "        print('  (No line limits changed.)')\n"
    "\n"
    "print('plot_decomp_results defined.')\n"
))

# 6 datasets header
cells.append(md("---\n## Section 1 · Dataset loading"))

# 7 load 14
cells.append(code(
    "DATA_14, idx_14, cvec_14, b0_14, u_init_14, p_init_14, on_t_14, off_t_14 = quick_setup(\n"
    '    os.path.join(DATA_DIR, "ieee14_enhanced.json"),\n'
    "    carbon_price=None, voll=20_000.0, slack_bus=None,\n"
    ")\n"
    'print(f"IEEE 14-bus  buses={DATA_14.nB}  lines={len(DATA_14.lines)}  gens={len(DATA_14.gens)}  T={int(DATA_14.T)}h")\n'
))

# 8 load 39
cells.append(code(
    "DATA_39, idx_39, cvec_39, b0_39, u_init_39, p_init_39, on_t_39, off_t_39 = quick_setup(\n"
    '    os.path.join(DATA_DIR, "ieee39_newengland.json"),\n'
    "    carbon_price=None, voll=20_000.0, slack_bus=None,\n"
    "    demand_scaling=1, ren_scaling=1.0,\n"
    ")\n"
    'print(f"IEEE 39-bus  buses={DATA_39.nB}  lines={len(DATA_39.lines)}  gens={len(DATA_39.gens)}  T={int(DATA_39.T)}h")\n'
))

# 9 load 57
cells.append(code(
    "DATA_57, idx_57, cvec_57, b0_57, u_init_57, p_init_57, on_t_57, off_t_57 = quick_setup(\n"
    '    os.path.join(DATA_DIR, "ieee57_uc_matpower.json"),\n'
    "    carbon_price=50.0, voll=500.0, slack_bus=0,\n"
    "    ren_scaling=1.0, demand_scaling=1.0,\n"
    ")\n"
    "u_init_57 = [0]*len(DATA_57.gens); p_init_57 = [0.0]*len(DATA_57.gens)\n"
    "on_t_57   = [0]*len(DATA_57.gens); off_t_57  = [0]*len(DATA_57.gens)\n"
    "# Warm-start G0 exactly as in bs_7grids.ipynb\n"
    "u_init_57[0]=1; p_init_57[0]=DATA_57.gens[0].Pmax; on_t_57[0]=DATA_57.gens[0].UT\n"
    'print(f"IEEE 57-bus  buses={DATA_57.nB}  lines={len(DATA_57.lines)}  gens={len(DATA_57.gens)}  T={int(DATA_57.T)}h")\n'
))

# 10 factual header
cells.append(md("---\n## Section 2 · Factual UC and emissions baseline"))

def factual_cell(g):
    N = g.upper()
    return code(
        f"_, solF_{g}, zF_{g} = solve_uc_with_cost_4b(\n"
        f"    data=DATA_{g}, idx=idx_{g}, cvec=cvec_{g},\n"
        f"    window_size=int(DATA_{g}.T), per_bus_neutrality=True,\n"
        f"    u_init=u_init_{g}, p_init=p_init_{g},\n"
        f"    on_time_init=on_t_{g}, off_time_init=off_t_{g}, output_flag=0,\n"
        f")\n"
        f"assert solF_{g} is not None\n"
        f"e_{g}         = np.array([float(gen.emission_rate) for gen in DATA_{g}.gens])\n"
        f"E_factual_{g} = float(np.sum(e_{g}[:, None] * solF_{g}['p']))\n"
        f"foil_fn_{g}   = make_emissions_foil_4b(DATA_{g}, alpha=ALPHA, E_factual=E_factual_{g})\n"
        f"b_free_idx_{g}, util_{g} = get_congested_free_lines(solF_{g}, b0_{g}, thr=UTIL_THR)\n"
        f"bL_{g}, bU_{g} = build_b_bounds(b0_{g}, b_free_idx_{g})\n"
        f"w_{g}          = make_line_weights(DATA_{g}, b0_{g}, util=util_{g})\n"
        f"oracle_{g}     = UCWeakWCEOracle(\n"
        f"    data=DATA_{g}, cvec=cvec_{g}, idx=idx_{g},\n"
        f"    window_size=int(DATA_{g}.T), per_bus_neutrality=True,\n"
        f"    u_init=u_init_{g}, p_init=p_init_{g}, on_t=on_t_{g}, off_t=off_t_{g},\n"
        f"    foil_extra_constr_fn=foil_fn_{g}, output_flag=0,\n"
        f")\n"
        f"vF_{g}, _, solFoil_b0_{g} = oracle_{g}.solve_foil(b0_{g})\n"
        f'print(f"IEEE {g.replace("_","-")}-bus  E_factual={{E_factual_{g}:.2f}} tCO2  '
        f'target={{(1-ALPHA)*E_factual_{g}:.2f}}  '
        f'foil_at_b0={{\'feas\' if solFoil_b0_{g} else \'infeas\'}}'
        f'  free={{len(b_free_idx_{g})}} lines")\n'
    )

cells.append(factual_cell("14"))

# IEEE 14: B&S adds foil_no_shed to prevent load-shedding in the foil (Cell 39 of bs_7grids)
cells.append(code(
    "foil_fn_14_base = foil_fn_14\n"
    "def foil_fn_14(m, var):\n"
    "    foil_fn_14_base(m, var)\n"
    "    m.addConstr(\n"
    "        gp.quicksum(var['shed'][b, t]\n"
    "                    for b in range(DATA_14.nB)\n"
    "                    for t in range(int(DATA_14.T))) == 0,\n"
    "        name='foil_no_shed')\n"
    "oracle_14.foil_extra_constr_fn = foil_fn_14\n"
    "oracle_14.cache_foil.clear()\n"
    "vF_14, _, solFoil_b0_14 = oracle_14.solve_foil(b0_14)\n"
    'print(f"IEEE 14 foil_no_shed: foil_at_b0={\'feas\' if solFoil_b0_14 else \'infeas\'}")\n'
))

cells.append(factual_cell("39"))
cells.append(factual_cell("57"))

# big-M header
cells.append(md("---\n## Section 3 · Big-M calibration\n\nEstimate `big_M_mu` as 10× the maximum LP dual on fmax/fmin constraints at b₀."))

# big-M cell
cells.append(code(
    "def estimate_big_M_mu(DATA, idx, cvec, b0, u_init, p_init, on_t, off_t, multiplier=10.0):\n"
    "    from uc_master_relax_4b import build_uc_relax_master_varfmax_4b\n"
    "    bL = b0 * 0.5; bU = b0 * 2.0\n"
    "    b_all = list(range(len(DATA.lines)))\n"
    "    m, var, bcap = build_uc_relax_master_varfmax_4b(\n"
    "        data=DATA, idx=idx, cvec=cvec,\n"
    "        window_size=int(DATA.T), per_bus_neutrality=True,\n"
    "        u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,\n"
    "        b0=b0, node_bL=bL, node_bU=bU, b_free_idx=b_all, output_flag=0,\n"
    "    )\n"
    "    set_objective_from_cvec(m, var, idx, cvec)\n"
    "    m.optimize()\n"
    "    max_pi = 0.0\n"
    "    if m.Status == 2:\n"
    "        for c in m.getConstrs():\n"
    "            nm = c.ConstrName\n"
    '            if nm.startswith("foil_fmax") or nm.startswith("fmax") or nm.startswith("fmin"):\n'
    "                try: max_pi = max(max_pi, abs(c.Pi))\n"
    "                except Exception: pass\n"
    "    m.dispose()\n"
    "    big_M = multiplier * max(max_pi, 1.0)\n"
    '    print(f"  max|dual_fmax|={max_pi:.4f}  big_M_mu={big_M:.2f}")\n'
    "    return big_M\n"
    "\n"
    "\n"
    'print("Calibrating big-M ...")\n'
    "big_M_mu_14 = estimate_big_M_mu(DATA_14, idx_14, cvec_14, b0_14, u_init_14, p_init_14, on_t_14, off_t_14)\n"
    "big_M_mu_39 = estimate_big_M_mu(DATA_39, idx_39, cvec_39, b0_39, u_init_39, p_init_39, on_t_39, off_t_39)\n"
    "big_M_mu_57 = estimate_big_M_mu(DATA_57, idx_57, cvec_57, b0_57, u_init_57, p_init_57, on_t_57, off_t_57)\n"
    'print(f"big_M_mu: 14={big_M_mu_14:.1f}  39={big_M_mu_39:.1f}  57={big_M_mu_57:.1f}")\n'
))

# DECOMP header
# ── helper generators ────────────────────────────────────────────────────────

def decomp_cell(g, big_M_mult=1.0, time_limit=300,
                master_out=0, max_iter=15, suffix="", comp_mode="sos1",
                use_bs_hint=True):
    """Generate one DECOMP run cell.

    use_bs_hint: if True, load b_hat from bs_{g}_checkpoint.json (if it exists)
    and pass it as b_hat_hint to DECOMP so the master always has a feasible
    incumbent after the first KKT cut is added.
    """
    var   = f"decomp_{g}{suffix}"
    res   = f"res_{g}{suffix}"
    ckpt  = f"decomp_{g}{suffix}_checkpoint.json"

    hint_block = ""
    if use_bs_hint:
        hint_block = (
            f'_bs_hint_{g} = None\n'
            f'_bs_ckpt_{g} = "bs_{g}_checkpoint.json"\n'
            f'if os.path.exists(_bs_ckpt_{g}):\n'
            f'    import json as _json\n'
            f'    with open(_bs_ckpt_{g}) as _fh:\n'
            f'        _bs_cp_{g} = _json.load(_fh)\n'
            f'    _bs_hint_{g} = np.array(_bs_cp_{g}["best_b"], float)\n'
            f'    print(f"[hint] B&S b_hat loaded for IEEE {g} '
            f'(F_bs={{_bs_cp_{g}[\'best_F\']:.4f}})")\n'
            f'else:\n'
            f'    print("[hint] No B&S checkpoint found — running without hint")\n'
        )

    return code(
        hint_block +
        f'\nfor _f in ["{ckpt}"]:\n'
        f"    if os.path.exists(_f): os.remove(_f)\n"
        f"\n"
        f"{var} = UCDecomp4b(\n"
        f"    oracle=oracle_{g}, data=DATA_{g}, idx=idx_{g}, cvec=cvec_{g},\n"
        f"    foil_extra_constr_fn=foil_fn_{g},\n"
        f"    b0=b0_{g}, b_bounds=(bL_{g}, bU_{g}), b_free_idx=b_free_idx_{g},\n"
        f"    big_M_mu=big_M_mu_{g},\n"
        f"    eps_weak=1e-3, eps_obj=1e-3, max_iter={max_iter},\n"
        f"    output_flag=0, verbose=True, w=w_{g},\n"
        f'    checkpoint_path="{ckpt}",\n'
        f"    big_M_multiplier={big_M_mult},\n"
        f"    master_time_limit={time_limit},\n"
        f"    master_output_flag={master_out},\n"
        f"    master_mip_gap=1e-4,\n"
        f'    comp_mode="{comp_mode}",\n'
        + (f"    b_hat_hint=_bs_hint_{g},\n" if use_bs_hint else "")
        + f")\n"
        f"{res} = {var}.run(\n"
        f"    window_size=int(DATA_{g}.T), per_bus_neutrality=True,\n"
        f"    u_init=u_init_{g}, p_init=p_init_{g},\n"
        f"    on_time_init=on_t_{g}, off_time_init=off_t_{g},\n"
        f")\n"
        f'print(f"\\nIEEE {g}-bus | comp={comp_mode} success={{{res}[\'success\']}} '
        f'F_opt={{{res}[\'F_opt\']:.4f}} iters={{{res}[\'iterations\']}} '
        f'cert={{{res}[\'certified\']}} gap={{{res}[\'gap\']:.4f}} '
        f'LB={{{res}[\'master_LB\']:.4f}}  patterns={{{res}[\'seen_patterns\']}}  '
        f'big_M_mult={big_M_mult}")\n'
    )

def debug_fix_b_cell(g, comp_mode="bigM"):
    """Self-consistent debug_fix_b: KKT block uses plain-optimal-at-b_hat."""
    return code(
        f'import json as _json\n'
        f'_cp_path = "bs_{g}_checkpoint.json"\n'
        f'if os.path.exists(_cp_path):\n'
        f'    with open(_cp_path) as _fh:\n'
        f'        _cp = _json.load(_fh)\n'
        f'    _b_bs = np.array(_cp["best_b"], float)\n'
        f'    print(f"[4a] IEEE {g}: B&S F_opt={{_cp[\'best_F\']:.4f}}")\n'
        f'    _tester = UCDecomp4b(\n'
        f'        oracle=oracle_{g}, data=DATA_{g}, idx=idx_{g}, cvec=cvec_{g},\n'
        f'        foil_extra_constr_fn=foil_fn_{g},\n'
        f'        b0=b0_{g}, b_bounds=(bL_{g}, bU_{g}), b_free_idx=b_free_idx_{g},\n'
        f'        big_M_mu=big_M_mu_{g}, output_flag=0, verbose=True, w=w_{g},\n'
        f'        big_M_multiplier=1.0, master_output_flag=0,\n'
        f'        comp_mode="{comp_mode}",\n'
        f'    )\n'
        f'    _dbg = _tester.debug_fix_b(\n'
        f'        b_test=_b_bs,\n'
        f'        window_size=int(DATA_{g}.T), per_bus_neutrality=True,\n'
        f'        u_init=u_init_{g}, p_init=p_init_{g},\n'
        f'        on_time_init=on_t_{g}, off_time_init=off_t_{g},\n'
        f'        iis_path=f"debug_fix_b_{g}_self.ilp",\n'
        f'    )\n'
        f'    print(f"[4a] IEEE {g} self-consistent (comp={comp_mode}): '
        f'feasible={{_dbg[\'feasible\']}}  obj={{_dbg[\'obj\']}}  '
        f'cut_slack={{_dbg.get(\'cut_slack\')}}")\n'
        f'    del _tester\n'
        f'else:\n'
        f'    print(f"[4a] No B&S checkpoint for IEEE {g} — skipping.")\n'
    )


def bs_preprocess_cell(g, voll, max_nodes=200):
    """Branch-and-Sandwich preprocessing: runs B&S if the checkpoint is missing.

    Output: bs_{g}_checkpoint.json with the heuristic CE (b_hat).
    This is the warm-start source that DECOMP needs to find any incumbent
    in Iter 2+ (the master MIP is infeasible-to-explore without one — see
    diagnostic in Section 4a / 4b.1b for evidence).

    Fallback chain:
      1. Use cached checkpoint if present.
      2. Run B&S to find a CE → save checkpoint.
      3. If B&S fails (no CE found), default to b_U (max expansion):
         it's foil-feasible by construction and almost always a CE,
         just with a large F.  Save synthetic checkpoint.
    """
    ck = f"bs_{g}_checkpoint.json"
    return code(
        f'_bs_ckpt = "{ck}"\n'
        f'if os.path.exists(_bs_ckpt):\n'
        f'    import json as _json\n'
        f'    with open(_bs_ckpt) as _fh:\n'
        f'        _cp = _json.load(_fh)\n'
        f'    print(f"[B&S] IEEE {g}: using cached checkpoint  "\n'
        f'          f"(F={{_cp[\'best_F\']:.4f}}, certified={{_cp.get(\'certified\')}})  "\n'
        f'          f"— delete {{_bs_ckpt!r}} to re-run.")\n'
        f'else:\n'
        f'    print(f"[B&S] IEEE {g}: no checkpoint — running B&S preprocessing  '
        f'(max_nodes={max_nodes}) ...")\n'
        f'    # Violation function: normalised emission excess + shed penalty\n'
        f'    def _make_viol_fn():\n'
        f'        E_f = E_factual_{g}; tgt = (1 - ALPHA) * E_f\n'
        f'        e_g = np.array([float(gen.emission_rate) for gen in DATA_{g}.gens])\n'
        f'        nB = DATA_{g}.nB; T = int(DATA_{g}.T)\n'
        f'        total_demand_MWh = float(np.sum(DATA_{g}.demand))\n'
        f'        def viol_fn(m, var, idx):\n'
        f'            s = m.addVar(lb=0.0, name="emis_viol")\n'
        f'            total_emis = gp.quicksum(e_g[g_] * var["p"][g_, t]\n'
        f'                                      for g_ in range(len(e_g)) for t in range(T))\n'
        f'            total_shed = gp.quicksum(var["shed"][b, t]\n'
        f'                                      for b in range(nB) for t in range(T))\n'
        f'            m.addConstr(\n'
        f'                s >= (total_emis - tgt) / max(E_f, 1.0)\n'
        f'                   + total_shed / max(total_demand_MWh, 1.0),\n'
        f'                name="emis_viol_constr",\n'
        f'            )\n'
        f'            m.update(); return s\n'
        f'        return viol_fn\n'
        f'    _viol_fn = _make_viol_fn()\n'
        f'    _bs_obj = UCBranchAndSandwichWCE_4b(\n'
        f'        oracle=oracle_{g},\n'
        f'        data=DATA_{g}, idx=idx_{g}, cvec=cvec_{g},\n'
        f'        foil_extra_constr_fn=foil_fn_{g},\n'
        f'        b0=b0_{g}, b_bounds=(bL_{g}, bU_{g}), b_free_idx=b_free_idx_{g},\n'
        f'        eps_b=1.0, eps_obj=1e-3, eps_weak=1e-3,\n'
        f'        max_nodes={max_nodes},\n'
        f'        relax_cost_ub=None, master_time_limit=None,\n'
        f'        output_flag=0, verbose=True, w=w_{g},\n'
        f'        foil_violation_expr_fn=_viol_fn,\n'
        f'        lagrange_penalty=500.0,\n'
        f'        checkpoint_path=_bs_ckpt,\n'
        f'    )\n'
        f'    _bs_res = _bs_obj.run(\n'
        f'        window_size=int(DATA_{g}.T), per_bus_neutrality=True,\n'
        f'        u_init=u_init_{g}, p_init=p_init_{g},\n'
        f'        on_t=on_t_{g}, off_t=off_t_{g},\n'
        f'        compute_final_mip_lb=False,\n'
        f'    )\n'
        f'    if _bs_res.get("success") and _bs_res.get("best_b") is not None:\n'
        f'        print(f"[B&S] IEEE {g}: done  F={{_bs_res[\'F_opt\']:.4f}}  "\n'
        f'              f"nodes={{_bs_res.get(\'nodes\')}}  certified={{_bs_res.get(\'certified\')}}")\n'
        f'    else:\n'
        f'        # B&S failed to find a CE — fall back to bU (max expansion),\n'
        f'        # which is foil-feasible by construction and almost always a CE.\n'
        f'        # F(bU) is large but the warm-start needs ANY valid CE, not the best.\n'
        f'        _bU = bU_{g}.copy()\n'
        f'        _vp_bU, _, _ = oracle_{g}.solve_plain(_bU)\n'
        f'        _vd_bU, _, _ = oracle_{g}.solve_foil(_bU)\n'
        f'        if _vp_bU is not None and _vd_bU is not None and _vd_bU <= _vp_bU + 1e-3:\n'
        f'            _F_bU = float(np.sum(w_{g}[b_free_idx_{g}] * np.abs(\n'
        f'                _bU[b_free_idx_{g}] - b0_{g}[b_free_idx_{g}])))\n'
        f'            import json as _json\n'
        f'            with open(_bs_ckpt, "w") as _fh:\n'
        f'                _json.dump({{\n'
        f'                    "best_b": _bU.tolist(),\n'
        f'                    "best_F": _F_bU,\n'
        f'                    "certified": False,\n'
        f'                    "fallback": "bU (max expansion); B&S found no CE",\n'
        f'                }}, _fh)\n'
        f'            print(f"[B&S] IEEE {g}: B&S failed — fell back to bU  "\n'
        f'                  f"(F={{_F_bU:.4f}}, v_foil={{_vd_bU:.2f}} ≤ v_plain={{_vp_bU:.2f}}, "\n'
        f'                  f"certified=False).  Saved synthetic checkpoint.")\n'
        f'        else:\n'
        f'            raise RuntimeError(\n'
        f'                f"IEEE {g}: neither B&S nor bU produced a CE — "\n'
        f'                f"v_foil(bU)={{_vd_bU}} > v_plain(bU)={{_vp_bU}} + 1e-3.  "\n'
        f'                f"Check problem setup (emissions target may be infeasible at any b)."\n'
        f'            )\n'
    )


def plot_decomp_cell(g):
    """Re-run plain/foil oracles at the DECOMP result's b_hat and plot."""
    return code(
        f'res = res_{g}\n'
        f'if not res.get("success") or res.get("b_hat") is None:\n'
        f'    print(f"[plot] IEEE {g}: no successful result — skipping plot.")\n'
        f'else:\n'
        f'    _bh = np.array(res["b_hat"], dtype=float)\n'
        f'    _, _, _sol_plain = oracle_{g}.solve_plain(_bh)\n'
        f'    _, _, _sol_foil  = oracle_{g}.solve_foil(_bh)\n'
        f'    plot_decomp_results(\n'
        f'        DATA_{g}, b0_{g}, _bh,\n'
        f'        sol0=solF_{g},\n'
        f'        sol_plain_hat=_sol_plain,\n'
        f'        sol_foil_hat=_sol_foil,\n'
        f'        res=res,\n'
        f'        label=f"IEEE {g}-bus",\n'
        f'        foil_label=f"emissions −{{ALPHA:.0%}}",\n'
        f'        method="DECOMP",\n'
        f'    )\n'
    )


def cross_pattern_test_cell(g, comp_mode="bigM"):
    """Cross-pattern debug_fix_b: extracts u_1 from Iter 1, tests at b_hat.

    This is the actual model the DECOMP run() searches at Iter 2 (with b fixed
    to b_hat).  Definitive test of whether the KKT block for u_1 admits the
    known CE point b_hat.
    """
    return code(
        f'import json as _json\n'
        f'_cp_path = "bs_{g}_checkpoint.json"\n'
        f'if os.path.exists(_cp_path):\n'
        f'    with open(_cp_path) as _fh:\n'
        f'        _cp = _json.load(_fh)\n'
        f'    _b_bs = np.array(_cp["best_b"], float)\n'
        f'    print(f"[4a-cross] IEEE {g}: B&S F_opt={{_cp[\'best_F\']:.4f}}")\n'
        f'    _tester = UCDecomp4b(\n'
        f'        oracle=oracle_{g}, data=DATA_{g}, idx=idx_{g}, cvec=cvec_{g},\n'
        f'        foil_extra_constr_fn=foil_fn_{g},\n'
        f'        b0=b0_{g}, b_bounds=(bL_{g}, bU_{g}), b_free_idx=b_free_idx_{g},\n'
        f'        big_M_mu=big_M_mu_{g}, output_flag=0, verbose=True, w=w_{g},\n'
        f'        big_M_multiplier=1.0, master_output_flag=0,\n'
        f'        comp_mode="{comp_mode}",\n'
        f'    )\n'
        f'    # Step 1: replicate Iter 1 of run() to get u_1\n'
        f'    _b_1, _u_1, _F_1 = _tester.iter1_pattern(\n'
        f'        window_size=int(DATA_{g}.T), per_bus_neutrality=True,\n'
        f'        u_init=u_init_{g}, p_init=p_init_{g},\n'
        f'        on_time_init=on_t_{g}, off_time_init=off_t_{g},\n'
        f'    )\n'
        f'    if _u_1 is None:\n'
        f'        print("[4a-cross] iter1_pattern failed — skipping.")\n'
        f'    else:\n'
        f'        # Step 2: test KKT block for u_1 with b fixed to b_hat\n'
        f'        _dbg = _tester.debug_fix_b(\n'
        f'            b_test=_b_bs, u_j_override=_u_1,\n'
        f'            window_size=int(DATA_{g}.T), per_bus_neutrality=True,\n'
        f'            u_init=u_init_{g}, p_init=p_init_{g},\n'
        f'            on_time_init=on_t_{g}, off_time_init=off_t_{g},\n'
        f'            iis_path=f"debug_fix_b_{g}_cross.ilp",\n'
        f'        )\n'
        f'        print(f"[4a-cross] IEEE {g} cross-pattern (comp={comp_mode}): '
        f'feasible={{_dbg[\'feasible\']}}  obj={{_dbg[\'obj\']}}  '
        f'cut_slack={{_dbg.get(\'cut_slack\')}}")\n'
        f'        print(f"           interpretation: tiny cut_slack ⇒ b_hat is a '
        f'corner of feasible region ⇒ B&B will struggle even though formulation '
        f'is correct.")\n'
        f'    del _tester\n'
        f'else:\n'
        f'    print(f"[4a-cross] No B&S checkpoint for IEEE {g} — skipping.")\n'
    )


# ── Section 4a · debug_fix_b diagnostics ─────────────────────────────────────
cells.append(md(
    "---\n## Section 4a · Diagnostic — `debug_fix_b` with B&S solution\n\n"
    "Two tests per grid, both with `b` fixed to the known CE `b_hat` from B&S "
    "and `comp_mode='bigM'` (matches Section 4b):\n\n"
    "1. **Self-consistent** — KKT block built for `u^* = plain-optimal-at-b_hat`. "
    "Verifies formulation correctness; should always be feasible since `b_hat` IS a CE.\n"
    "2. **Cross-pattern** — KKT block built for `u_1` (the pattern DECOMP discovers "
    "at Iter 1, from a *different* `b_1`).  This is the actual model DECOMP's "
    "run() solves at Iter 2 with `b` fixed.\n\n"
    "**`cut_slack`** = `LP_cost(u^j, b_hat) − foil_cost` at the solution.  "
    "Tiny slack (< 1e-2) ⇒ the optimality cut is nearly binding at `b_hat`, so the "
    "feasible region is essentially a single point — B&B has to land on it exactly, "
    "which explains why Section 4b times out even though the formulation is correct."
))
cells.append(md("### 4a.1 · IEEE 14-bus — self-consistent"))
cells.append(debug_fix_b_cell("14", comp_mode="bigM"))
cells.append(md("### 4a.2 · IEEE 14-bus — cross-pattern (u_1 at b_hat)"))
cells.append(cross_pattern_test_cell("14", comp_mode="bigM"))
cells.append(md("### 4a.3 · IEEE 39-bus — self-consistent"))
cells.append(debug_fix_b_cell("39", comp_mode="bigM"))
cells.append(md("### 4a.4 · IEEE 39-bus — cross-pattern"))
cells.append(cross_pattern_test_cell("39", comp_mode="bigM"))
cells.append(md("### 4a.5 · IEEE 57-bus — self-consistent"))
cells.append(debug_fix_b_cell("57", comp_mode="bigM"))
cells.append(md("### 4a.6 · IEEE 57-bus — cross-pattern"))
cells.append(cross_pattern_test_cell("57", comp_mode="bigM"))

# ── Section 4b · Full pipeline: B&S preprocess → DECOMP → plot ────────────────
cells.append(md(
    "---\n## Section 4b · Full self-contained pipeline\n\n"
    "Per grid: **B&S preprocessing** (runs only if `bs_<grid>_checkpoint.json` is "
    "missing) → **DECOMP** with that B&S CE as the warm-start hint → "
    "**plot** the resulting CE.\n\n"
    "Why B&S is required as a preprocessor: the DECOMP master MIP at Iter 2+ has a "
    "bigM-complementarity structure where Gurobi's default heuristics cannot "
    "find any integer-feasible solution (demonstrated in 4b.1b cold-start: 67K "
    "B&B nodes, 0 incumbents in 900s).  B&S only solves LPs (no MIP), runs in "
    "minutes, and produces a heuristic CE that DECOMP uses to bootstrap.  After "
    "the first checkpoint exists the B&S cell is a fast no-op."
))

# IEEE 14
cells.append(md("### 4b.1 · IEEE 14-bus"))
cells.append(md("**Step 1/3** — B&S preprocessing (skipped if checkpoint exists)"))
cells.append(bs_preprocess_cell("14", voll=20000.0, max_nodes=200))
cells.append(md("**Step 2/3** — DECOMP refinement (warm-started from B&S)"))
cells.append(decomp_cell("14", big_M_mult=1.0, time_limit=900,
                          master_out=1, max_iter=15, suffix="",
                          comp_mode="bigM"))
cells.append(md("**Step 3/3** — Plot result"))
cells.append(plot_decomp_cell("14"))

# Cold-start ablation (keep for reference; expected to fail at Iter 2)
cells.append(md(
    "### 4b.1b · IEEE 14-bus — ABLATION: NO warm-start (cold start)\n\n"
    "Same as 4b.1 but `use_bs_hint=False`: no B&S CE, no `F ≤ F_hint`, no "
    "analytic warm-start.  Expected to halt after Iter 1 with `success=False` "
    "and `termination_reason='time_limit_no_incumbent'` — empirical evidence "
    "that the warm-start is structurally required, not an optimization shortcut."
))
cells.append(decomp_cell("14", big_M_mult=1.0, time_limit=900,
                          master_out=1, max_iter=15, suffix="_nohint",
                          comp_mode="bigM", use_bs_hint=False))

# IEEE 39
cells.append(md("### 4b.2 · IEEE 39-bus"))
cells.append(md("**Step 1/3** — B&S preprocessing"))
cells.append(bs_preprocess_cell("39", voll=20000.0, max_nodes=200))
cells.append(md("**Step 2/3** — DECOMP refinement"))
cells.append(decomp_cell("39", big_M_mult=1.0, time_limit=900,
                          master_out=1, max_iter=15, suffix="",
                          comp_mode="bigM"))
cells.append(md("**Step 3/3** — Plot result"))
cells.append(plot_decomp_cell("39"))

# IEEE 57
cells.append(md("### 4b.3 · IEEE 57-bus"))
cells.append(md("**Step 1/3** — B&S preprocessing"))
cells.append(bs_preprocess_cell("57", voll=500.0, max_nodes=200))
cells.append(md("**Step 2/3** — DECOMP refinement"))
cells.append(decomp_cell("57", big_M_mult=1.0, time_limit=900,
                          master_out=1, max_iter=15, suffix="",
                          comp_mode="bigM"))
cells.append(md("**Step 3/3** — Plot result"))
cells.append(plot_decomp_cell("57"))

# ── Section 4c · LB-stagnation Fix 1 (hybrid bigM + indicator) ────────────────
cells.append(md(
    "---\n## Section 4c · Fix 1: `comp_mode=\"hybrid\"` (LB-stagnation experiment)\n\n"
    "Same B&S warm-start pipeline as 4b, but with `comp_mode=\"hybrid\"`: bigM is "
    "kept only for free-line flow-limit complementarity pairs (whose slack "
    "`b[ell] ± f^j` involves a master decision variable), and **all other** "
    "complementarity pairs (gen bounds, ramps, shed, curt, shift) are encoded "
    "as Gurobi indicator constraints.  Indicators are enforced by B&B branching "
    "and **do not** linearise to a loose big-M in the LP relaxation, so each "
    "new KKT block actually tightens `ObjBound` instead of being absorbed by "
    "fractional `z`s.  See `DECOMP_lb_stagnation.md` Fix 1.\n\n"
    "Per-iteration `[LB diag]` lines (always printed in verbose mode) report "
    "`ObjBound` (CCG lower bound), `ObjVal` (incumbent upper bound), `MIPGap`, "
    "and the freshly solved Root LP value from `m.relax()`.  Compare against "
    "4b: in bigM mode `ObjBound ≈ Root LP ≈ constant` across iterations; with "
    "hybrid we expect `ObjBound` to grow each iteration.\n\n"
    "Results are stored in `res_14_hybrid`, `res_39_hybrid`, `res_57_hybrid` "
    "(distinct from the bigM `res_<g>` baselines)."
))

# IEEE 14 hybrid
cells.append(md("### 4c.1 · IEEE 14-bus — hybrid"))
cells.append(decomp_cell("14", big_M_mult=1.0, time_limit=900,
                          master_out=1, max_iter=15, suffix="_hybrid",
                          comp_mode="hybrid"))

# IEEE 39 hybrid
cells.append(md("### 4c.2 · IEEE 39-bus — hybrid"))
cells.append(decomp_cell("39", big_M_mult=1.0, time_limit=900,
                          master_out=1, max_iter=15, suffix="_hybrid",
                          comp_mode="hybrid"))

# IEEE 57 hybrid
cells.append(md("### 4c.3 · IEEE 57-bus — hybrid"))
cells.append(decomp_cell("57", big_M_mult=1.0, time_limit=900,
                          master_out=1, max_iter=15, suffix="_hybrid",
                          comp_mode="hybrid"))

# Side-by-side bigM vs hybrid summary
cells.append(md(
    "### 4c.4 · Comparison summary (bigM vs hybrid)\n"
    "Side-by-side per-grid comparison of certified gap and final LB.  "
    "Hybrid should show strictly higher (or equal) `master_LB` and smaller `gap_pct`."
))
cells.append(code(
    "def _fmt(r):\n"
    "    if r is None or not r.get('success'):\n"
    "        return f\"{'N/A':>10} {'N/A':>10} {'N/A':>8}\"\n"
    "    return (f\"{r['F_opt']:>10.4f} {r['master_LB']:>10.4f} \"\n"
    "            f\"{r['gap_pct']:>7.2f}%\")\n"
    "\n"
    "print(f\"{'':<14} | {'=== bigM ===':^29} | {'=== hybrid ===':^29}\")\n"
    "print(f\"{'Grid':<14} | {'F_opt':>10} {'LB':>10} {'gap%':>8} | \"\n"
    "      f\"{'F_opt':>10} {'LB':>10} {'gap%':>8}\")\n"
    "print('-' * 78)\n"
    "for label, r_bm, r_hy in [\n"
    "    ('IEEE 14-bus', globals().get('res_14'), globals().get('res_14_hybrid')),\n"
    "    ('IEEE 39-bus', globals().get('res_39'), globals().get('res_39_hybrid')),\n"
    "    ('IEEE 57-bus', globals().get('res_57'), globals().get('res_57_hybrid')),\n"
    "]:\n"
    "    print(f'{label:<14} | {_fmt(r_bm)} | {_fmt(r_hy)}')\n"
))

# Results header
cells.append(md("---\n## Section 5 · Results summary"))

# Summary table
cells.append(code(
    'print(f"{chr(34)*0}{\"Grid\":<14} {\"F_opt\":>10} {\"Iters\":>7} {\"Certified\":>11} {\"Gap%\":>8} {\"Patterns\":>10}")\n'
    'print("-" * 64)\n'
    'for label, res in [("IEEE 14-bus", res_14), ("IEEE 39-bus", res_39), ("IEEE 57-bus", res_57)]:\n'
    '    cert = "YES" if res["certified"] else "NO"\n'
    '    Fopt = f"{res[\'F_opt\']:.4f}" if res["success"] else "N/A"\n'
    '    gapc = f"{res[\'gap_pct\']:.2f}%" if res["success"] else "N/A"\n'
    '    print(f"{label:<14} {Fopt:>10} {res[\'iterations\']:>7} {cert:>11} {gapc:>8} {res[\'seen_patterns\']:>10}")\n'
))

# CE verification
cells.append(code(
    'print("\\nCE verification (v_foil <= v_plain + eps):")\n'
    'EPS = 1e-3\n'
    'for label, res, oracle in [\n'
    '    ("IEEE 14-bus", res_14, oracle_14),\n'
    '    ("IEEE 39-bus", res_39, oracle_39),\n'
    '    ("IEEE 57-bus", res_57, oracle_57),\n'
    ']:\n'
    '    if not res["success"]:\n'
    '        print(f"  {label}: no CE found"); continue\n'
    '    vp, _, _ = oracle.solve_plain(res["b_hat"])\n'
    '    vd, _, _ = oracle.solve_foil(res["b_hat"])\n'
    '    ok = vd is not None and vp is not None and vd <= vp + EPS\n'
    '    print(f"  {label}: v_plain={vp:.2f}  v_foil={vd:.2f}  CE_ok={ok}")\n'
))

nb_json = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = os.path.join(os.path.dirname(__file__), "decomp_3grids.ipynb")
with open(out, "w") as f:
    json.dump(nb_json, f, indent=1)
print(f"Written: {out}  ({len(cells)} cells)")
