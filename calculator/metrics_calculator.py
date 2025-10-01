
"""
metrics_calculator 
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

@dataclass
class RebaseRecord:
    year: int
    prev_eom_ts: pd.Timestamp
    prev_eom_nav: float
    next_first_ts: pd.Timestamp
    next_first_nav: float
    delta_abs: float           # next_first_nav - prev_eom_nav (currency units; negative => cash-out)
    factor_s: float            # scaling to chain-link: prev_eom_nav / next_first_nav
    cashflow_ccy: float        # prev_eom_nav - next_first_nav (positive => cash-out, negative => cash-in)

def _detect_year_boundary_rebases_absolute(equity_df: pd.DataFrame, tol_abs_ccy: float = 10.0) -> List[RebaseRecord]:
    """
    Detect cash-out / cash-in between the very last tick of December (year Y)
    and the very first tick of January (year Y+1) using an absolute currency threshold.
    Only Dec->Jan boundaries are considered. Returns sorted list of RebaseRecord by year.
    """
    if equity_df is None or equity_df.empty:
        return []
    df = equity_df.copy()
    # Ensure datetime (UTC) and sorted
    try:
        import pandas.api.types as ptypes
        if not (ptypes.is_datetime64_any_dtype(df["time_utc"]) or ptypes.is_datetime64tz_dtype(df["time_utc"])):
            df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    except Exception:
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc", "capital_ccy"]).sort_values("time_utc").reset_index(drop=True)
    df["year"] = df["time_utc"].dt.year

    years = sorted(df["year"].unique().tolist())
    records: List[RebaseRecord] = []

    for y in years:
        y_next = y + 1
        if y_next not in df["year"].values:
            continue
        part_prev = df[df["year"] == y]
        part_next = df[df["year"] == y_next]
        if part_prev.empty or part_next.empty:
            continue

        # last tick of year y, first tick of year y+1
        idx_prev = part_prev["time_utc"].idxmax()
        idx_next = part_next["time_utc"].idxmin()
        prev_ts = df.at[idx_prev, "time_utc"]
        next_ts = df.at[idx_next, "time_utc"]
        prev_nav = float(df.at[idx_prev, "capital_ccy"])
        next_nav = float(df.at[idx_next, "capital_ccy"])

        delta_abs = next_nav - prev_nav  # absolute difference in currency (signed)
        if abs(delta_abs) < float(tol_abs_ccy):
            continue  # treat as noise

        if next_nav <= 0.0:  # avoid invalid scaling
            continue

        factor_s = prev_nav / next_nav
        cashflow_ccy = prev_nav - next_nav  # positive => cash-out

        records.append(
            RebaseRecord(
                year=int(y),
                prev_eom_ts=prev_ts,
                prev_eom_nav=prev_nav,
                next_first_ts=next_ts,
                next_first_nav=next_nav,
                delta_abs=delta_abs,
                factor_s=factor_s,
                cashflow_ccy=cashflow_ccy,
            )
        )
    records.sort(key=lambda r: r.year)
    return records


# ---------- Imports ----------
import math
import re
from dataclasses import dataclass, field 
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path
from typing import List, Optional, Tuple
import json
import fnmatch

import numpy as np
import pandas as pd

# === Monte Carlo helpers (stationary bootstrap / moving block bootstrap) ===
def _compose_year_month_yyyy_mm(y, m):
    try:
        y = int(y); m = int(m)
        if 1 <= m <= 12:
            return f"{y:04d}-{m:02d}"
    except Exception:
        pass
    return None

def _apply_rebases_to_equity(equity_df: pd.DataFrame, rebases: List[RebaseRecord]) -> pd.DataFrame:
    """
    Apply chain-link scaling to remove Dec→Jan cash-flow jumps.
    For each boundary record, multiply all rows with time_utc >= next_first_ts by factor_s.
    Rebases are applied in chronological order; factors thus accumulate.
    Returns a new DataFrame (does not mutate the input).
    """
    if equity_df is None or equity_df.empty or not rebases:
        return equity_df.copy() if equity_df is not None else pd.DataFrame()

    df = equity_df.copy()
    # Ensure datetime (UTC) and sorted ascending by time
    try:
        import pandas.api.types as ptypes
        if not (ptypes.is_datetime64_any_dtype(df["time_utc"]) or ptypes.is_datetime64tz_dtype(df["time_utc"])):
            df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    except Exception:
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")

    df = df.dropna(subset=["time_utc", "capital_ccy"]).sort_values("time_utc").reset_index(drop=True)

    # Apply each boundary scale in chronological order
    for rec in sorted(rebases, key=lambda r: r.year):
        mask = df["time_utc"] >= rec.next_first_ts
        if mask.any():
            df.loc[mask, "capital_ccy"] = df.loc[mask, "capital_ccy"] * float(rec.factor_s)

    return df


def _yearly_raw_nav_bounds(equity_df_raw: "pd.DataFrame") -> dict:
    """
    For each calendar year Y present in raw equity EoM series:
      - year_start_nav_ccy_raw[Y] = last available EoM NAV in year (Y-1)
      - year_end_nav_ccy_raw[Y]   = last available EoM NAV in year Y
    Uses raw (pre-rebase) EoM NAV to reflect factual account value.
    Returns dict: Y -> {"start": float|nan, "end": float|nan}
    """
    import pandas as pd
    import numpy as np

# --- Unified sign classification settings (added in V2) ---
# Tie the sign epsilon to the display precision of monthly returns.
# If monthly returns are shown with 4 decimals, eps = 0.5 * 10**(-4) = 5e-5.
try:
    # If code defines MONTHLY_DECIMALS elsewhere, use it; otherwise default to 4.
    MONTHLY_DECIMALS  # type: ignore[name-defined]
    _SIGN_DECIMALS = int(MONTHLY_DECIMALS)  # noqa: F821
except Exception:
    _SIGN_DECIMALS = 4

_SIGN_EPS = 0.5 * (10.0 ** (-_SIGN_DECIMALS))

def _classify_sign_masks(arr, eps=_SIGN_EPS):
    """Return boolean masks (pos, neg, zero) using a consistent epsilon.

    pos: arr > +eps

    neg: arr < -eps

    zero: |arr| <= eps

    """
    import numpy as _np  # local import to avoid polluting module namespace
    a = _np.asarray(arr, dtype=float)
    pos = a > +eps
    neg = a < -eps
    zero = _np.abs(a) <= eps
    return pos, neg, zero

def _count_pos_neg_zero(arr, eps=_SIGN_EPS):
    """Return tuple (pos_count, neg_count, zero_count) using unified sign eps."""
    pos, neg, zero = _classify_sign_masks(arr, eps=eps)
    return int(pos.sum()), int(neg.sum()), int(zero.sum())
# --- End unified sign classification settings (added in V2) ---

    if equity_df_raw is None or len(equity_df_raw)==0:
        return {}
    try:
        eom_raw = _eom_nav_from_equity_no_sort(equity_df_raw)  # pd.Series with PeriodIndex (M)
    except Exception:
        return {}
    if eom_raw is None or len(eom_raw)==0:
        return {}
    # Build mapping
    out = {}
    years = sorted(set(eom_raw.index.year.tolist()))
    for y in years:
        # end = last EoM within year y
        end_vals = eom_raw[eom_raw.index.year == y]
        end_nav = float(end_vals.iloc[-1]) if len(end_vals) else float("nan")
        # start = last EoM within year y-1
        prev_vals = eom_raw[eom_raw.index.year == (y-1)]
        start_nav = float(prev_vals.iloc[-1]) if len(prev_vals) else float("nan")
        out[int(y)] = {"start": start_nav, "end": end_nav}
    return out


def _map_opening_cashflow_by_year(rebases: "list[RebaseRecord]") -> dict:
    """
    Build mapping: year -> opening cash flow (currency), where
    opening_cashflow_ccy(year Y) = cashflow_ccy at boundary (Y-1 -> Y).
    Positive values denote cash-out (withdrawal), negative values denote cash-in.
    """
    _map = {}
    if not rebases:
        return _map
    for r in rebases:
        try:
            _map[int(r.year) + 1] = float(r.cashflow_ccy)
        except Exception:
            continue
    return _map


def _compute_flow_aware_full_period_fields(equity_df_raw: "pd.DataFrame", rebases: "list[RebaseRecord]", starting_nav: float) -> dict:
    """
    Compute investor flow-aware money fields on the raw equity NAV (pre-rebase) plus cash flows from rebases.
    Returns a dict with fields to append to full_period_summary:
      - total_cash_out_ccy
      - total_cash_in_ccy
      - net_flows_ccy
      - observed_last_eom_nav_ccy
      - flow_aware_ending_value_ccy
      - flow_aware_wealth_multiple
    Rounding is applied to match typical CSV monetary fields: currency ~2 decimals (safe), multiple ~4 decimals.
    """
    import pandas as pd
    if equity_df_raw is None or len(equity_df_raw)==0:
        return {
            "total_cash_out_ccy": 0.0,
            "total_cash_in_ccy": 0.0,
            "net_flows_ccy": 0.0,
            "observed_last_eom_nav_ccy": float("nan"),
            "flow_aware_ending_value_ccy": float("nan"),
            "flow_aware_wealth_multiple": float("nan"),
        }
    # derive EoM of raw equity and take the very last available
    try:
        eom_raw = _eom_nav_from_equity_no_sort(equity_df_raw)
        observed_last_eom_nav = float(eom_raw.iloc[-1])
    except Exception:
        # fallback: last NAV of raw equity
        observed_last_eom_nav = float(equity_df_raw.sort_values("time_utc")["capital_ccy"].iloc[-1])

    total_cash_out = 0.0
    total_cash_in = 0.0
    if rebases:
        for r in rebases:
            if r.cashflow_ccy > 0:
                total_cash_out += float(r.cashflow_ccy)
            elif r.cashflow_ccy < 0:
                total_cash_in += float(-r.cashflow_ccy)

    flow_aware_ending_value = observed_last_eom_nav + total_cash_out - total_cash_in
    # Protect against division by zero
    fam = flow_aware_ending_value / float(starting_nav) if float(starting_nav) != 0.0 else float("nan")

    # rounding: money to 2 decimals; multiple to 4 decimals (consistent with other multiples)
    return {
        "total_cash_out_ccy": round(total_cash_out, 2),
        "total_cash_in_ccy": round(total_cash_in, 2),
        "net_flows_ccy": round(total_cash_in - total_cash_out, 2),
        "observed_last_eom_nav_ccy": round(observed_last_eom_nav, 2),
        "flow_aware_ending_value_ccy": round(flow_aware_ending_value, 2),
        "flow_aware_wealth_multiple": round(fam, 4),
    }



def write_rebasing_applied_csv(rebases: List[RebaseRecord], csv_path: Path) -> None:
    """
    Write an audit log of applied Dec→Jan cash-flow chain-link boundaries.
    Timestamps are NOT stored; only year and YYYY-MM are written (no exact date/time).
    Columns:
      year, prev_eom_month, prev_eom_nav, next_first_month, next_first_nav, delta_abs, factor_s, cashflow_ccy
    """
    import pandas as _pd
    cols = ["year","prev_eom_month","prev_eom_nav","next_first_month","next_first_nav","delta_abs","factor_s","cashflow_ccy"]
    if not rebases:
        _pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
        return

    def _ym(ts):
        try:
            return _pd.Timestamp(ts).strftime("%Y-%m")
        except Exception:
            return ""

    rows = []
    for r in rebases:
        rows.append({
            "year": int(r.year),
            "prev_eom_month": _ym(r.prev_eom_ts),
            "prev_eom_nav": float(r.prev_eom_nav),
            "next_first_month": _ym(r.next_first_ts),
            "next_first_nav": float(r.next_first_nav),
            "delta_abs": float(r.delta_abs),
            "factor_s": float(r.factor_s),
            "cashflow_ccy": float(r.cashflow_ccy),
        })
    df = _pd.DataFrame(rows, columns=cols)

    # Rounding
    df["year"] = df["year"].astype(int)
    df["factor_s"] = df["factor_s"].astype(float).round(6)
    for col in ["prev_eom_nav","next_first_nav","delta_abs","cashflow_ccy"]:
        df[col] = df[col].astype(float).round(2)

    df.to_csv(csv_path, index=False)


def _fmt_month_yyyy_mm(x):
    """Return 'YYYY-MM' for pandas Period/Timestamp/str 'YYYY[-MM[-DD]]'; otherwise None.
    """
    try:
        if isinstance(x, pd.Period):
            return f"{x.year:04d}-{x.month:02d}"
        if isinstance(x, pd.Timestamp):
            return f"{x.year:04d}-{x.month:02d}"
    except Exception:
        pass
    s = str(x) if x is not None else ""
    # accept 'YYYY-MM' or 'YYYY-MM-DD' or 'YYYY/MM' variants

    m = re.match(r"^(\d{4})[-/](\d{2})(?:[-/]\d{2})?$", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    # last resort: try toparse with pandas if available
    try:
        ts = pd.to_datetime(s)
        return f"{ts.year:04d}-{ts.month:02d}"
    except Exception:
        return None

def _mc_sanitize_Ls(Ls):
    if isinstance(Ls, int):
        Ls = [Ls]
    try:
        vals = sorted({int(x) for x in Ls if 3 <= int(x) <= 12})
    except Exception:
        vals = []
    return vals or list(range(3,13))

def _mc_parse_list_str(s: str, allow_full=False):
    if s is None or s == "":
        return None
    items = [x.strip() for x in str(s).split(",")]
    out = []
    for x in items:
        if allow_full and x.lower() in ("full", "full_period", "full-period"):
            out.append("full_period")
        else:
            try:
                out.append(int(x))
            except Exception:
                continue
    return out or None

def _mc_defaults_from_schema() -> dict:
    raw = _SCHEMA_CACHE.get("raw_schema", {})
    parms = (raw.get("parameters", {}) or {}).get("monte_carlo", {}) or {}
    rtime = (raw.get("runtime", {}) or {}).get("monte_carlo", {}) or {}
    method = rtime.get("method") or parms.get("method") or "stationary_bootstrap"
    L_raw = rtime.get("block_mean_length_months", parms.get("block_mean_length_months", None))
    if L_raw is None:
        L = list(range(3,13))
    else:
        if isinstance(L_raw, (list, tuple)):
            L = _mc_sanitize_Ls(L_raw)
        else:
            L = _mc_sanitize_Ls([L_raw])
    horizons = rtime.get("horizons_months") or parms.get("horizons_months") or [12, 36, "full_period"]
    n_paths = rtime.get("n_paths") or parms.get("n_paths") or 10000
    seed = rtime.get("seed") if rtime.get("seed", None) is not None else parms.get("seed", None)
    if seed is None:
        seed = 42  # default per methodology
    dd_mode = rtime.get("dd_thresholds_eom_mode") or parms.get("dd_thresholds_eom_mode") or "both"
    dd_thresholds = rtime.get("dd_thresholds_eom") or parms.get("dd_thresholds_eom") or [0.05, 0.07, 0.10, 0.20, 0.30]
    return {
        "method": method,
        "block_mean_length_months": L,
        "horizons_months": horizons,
        "n_paths": n_paths,
        "seed": seed,
        "dd_thresholds_eom_mode": dd_mode,
        "dd_thresholds_eom": dd_thresholds,
        "k_risk_levels": [5, 7, 10],
    }

def _stationary_bootstrap_indices(n: int, L: int, h: int, rng):
    p = 1.0 / max(1, int(L))
    idx = np.empty(h, dtype=int)
    start = rng.integers(0, n)
    offset = 0
    for t in range(h):
        if t == 0 or (rng.random() < p):
            start = rng.integers(0, n)
            offset = 0
        idx[t] = (start + offset) % n
        offset += 1
    return idx

def _moving_block_bootstrap_indices(n: int, L: int, h: int, rng):
    L = max(1, int(L))
    idx = []
    while len(idx) < h:
        s = rng.integers(0, n)
        block = [(s + j) % n for j in range(L)]
        idx.extend(block)
    return np.array(idx[:h], dtype=int)

def _simulate_mc_paths(monthly_returns, method: str, L: int, n_paths: int, h: int, seed: int|None):

    rng = np.random.default_rng(seed)
    n = len(monthly_returns)
    out = np.empty((n_paths, h), dtype=float)
    for i in range(n_paths):
        if method == "moving_block_bootstrap":
            idc = _moving_block_bootstrap_indices(n, L, h, rng)
        else:
            idc = _stationary_bootstrap_indices(n, L, h, rng)
        out[i, :] = monthly_returns[idc]
    return out

def _path_nav(r, nav0=100000.0):

    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        return nav0 * np.cumprod(1.0 + r)
    else:
        return nav0 * np.cumprod(1.0 + r, axis=1)

def _maxdd_eom(nav):

    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return abs(np.min(dd))

def _first_ttb_ge_kx_risk(nav, k: float, risk_pct: float):

    if risk_pct <= 0:
        return None
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    xr = abs(dd) / risk_pct
    hit = np.nonzero(xr >= float(k))[0]
    if len(hit) == 0:
        return None
    return int(hit[0] + 1)  # months are 1-indexed to match human-readable


def _compute_mc_summary_for(monthly_returns, method: str, L: int, h: int, n_paths: int, seed: int|None, risk_pct: float, dd_mode: str = "both", dd_thresholds: list[float] | None = None, nav0: float | None = None):
    if nav0 is None:
        try:
            nav0 = Config().starting_nav
        except Exception:
            nav0 = 100000.0

 
    paths = _simulate_mc_paths(monthly_returns, method, L, n_paths, h, seed)
    navs = _path_nav(paths, nav0=nav0)  # shape (n_paths, h)
    wealth_mult = navs[:, -1] / nav0
    cagr = np.power(wealth_mult, 12.0 / float(h)) - 1.0
    maxdd_mag = np.array([_maxdd_eom(navs[i]) for i in range(n_paths)])

    # Probabilities
    prob_neg_ret = float(np.mean(wealth_mult - 1.0 < 0.0))

    # Absolute EoM thresholds (in decimals)
    abs_probs = {}
    if dd_thresholds is None:
        dd_thresholds = [0.05, 0.07, 0.10, 0.20, 0.30]
    if dd_mode in ("absolute", "both"):
        for th in dd_thresholds:
            try:
                thf = float(th)
            except Exception:
                continue
            key = f"prob_maxdd_ge_{int(round(thf*100))}pc_eom"
            abs_probs[key] = float(np.mean(maxdd_mag >= thf))

    # ×Risk metrics
    k_levels = [5.0, 7.0, 10.0]
    prob_ge_k = {}
    prob_no_ge_k = {}
    cond_es_ge_k = {}
    ttb = {k: [] for k in k_levels}
    if dd_mode in ("normalized", "both"):
        xr = maxdd_mag / max(risk_pct, 1e-12)
        prob_ge_k = {k: float(np.mean(xr >= k)) for k in k_levels}
        prob_no_ge_k = {k: (1.0 - prob_ge_k[k]) for k in k_levels}
        for k in k_levels:
            mask = (xr >= k)
            cond_es_ge_k[k] = float(np.mean(xr[mask])) if np.any(mask) else float("nan")
        # time-to-breach (median months amongst paths that breach)
        for i in range(n_paths):
            nav = navs[i]
            for k in k_levels:
                t = _first_ttb_ge_kx_risk(nav, k, risk_pct)
                if t is not None:
                    ttb[k].append(t)
    ttb_p50 = {k: (float(np.median(ttb[k])) if len(ttb[k]) else float("nan")) for k in k_levels}

    # --- Policy-aware MC (Variant A: NAV cap) ---
    policy_agg = None
    try:
        cfg = Config()
        if getattr(cfg, "mc_policy", "none") == "cap_nav" and float(getattr(cfg, "mc_cap_nav_ccy", 0.0)) > 0.0:
            stats_per_path = []
            cap_nav = float(cfg.mc_cap_nav_ccy)
            apply_mode = str(getattr(cfg, "mc_policy_apply", "yearly"))
            start_month = int(getattr(cfg, "mc_policy_start_month", 1))
            tol = float(getattr(cfg, "mc_cap_tolerance_ccy", 0.0))
            for i in range(n_paths):
                stats = _mc_apply_policy_cap_nav(
                    path_returns=paths[i],
                    starting_nav=float(nav0),
                    cap_nav=cap_nav,
                    apply_mode=apply_mode,
                    start_month=start_month,
                    tolerance=tol,
                )
                stats_per_path.append(stats)
            policy_agg = _mc_aggregate_policy(stats_per_path)
    except Exception:
        policy_agg = None

    # Percentiles
    p05 = 5.0; p50 = 50.0; p95 = 95.0; p99 = 99.0
    out = {
        "ending_nav_p05": float(np.percentile(navs[:, -1], p05)),
        "ending_nav_p50": float(np.percentile(navs[:, -1], p50)),
        "ending_nav_p95": float(np.percentile(navs[:, -1], p95)),
        "wealth_multiple_p05": float(np.percentile(wealth_mult, p05)),
        "wealth_multiple_p50": float(np.percentile(wealth_mult, p50)),
        "wealth_multiple_p95": float(np.percentile(wealth_mult, p95)),
        "cagr_annualized_p05": float(np.percentile(cagr, p05)),
        "cagr_annualized_p50": float(np.percentile(cagr, p50)),
        "cagr_annualized_p95": float(np.percentile(cagr, p95)),
        "max_drawdown_magnitude_p50": float(np.percentile(maxdd_mag, p50)),
        "max_drawdown_magnitude_p95": float(np.percentile(maxdd_mag, p95)),
        "max_drawdown_magnitude_p99": float(np.percentile(maxdd_mag, p99)),
        "prob_negative_horizon_return": prob_neg_ret,
    }
    out.update(abs_probs)
    if dd_mode in ("normalized", "both"):
        out.update({
            "prob_maxdd_ge_5x_risk_eom": prob_ge_k.get(5.0, float("nan")),
            "prob_maxdd_ge_7x_risk_eom": prob_ge_k.get(7.0, float("nan")),
            "prob_maxdd_ge_10x_risk_eom": prob_ge_k.get(10.0, float("nan")),
            "prob_no_breach_ge_5x_risk_eom": prob_no_ge_k.get(5.0, float("nan")),
            "prob_no_breach_ge_7x_risk_eom": prob_no_ge_k.get(7.0, float("nan")),
            "prob_no_breach_ge_10x_risk_eom": prob_no_ge_k.get(10.0, float("nan")),
            "cond_es_maxdd_ge_5x_risk": cond_es_ge_k.get(5.0, float("nan")),
            "cond_es_maxdd_ge_7x_risk": cond_es_ge_k.get(7.0, float("nan")),
            "cond_es_maxdd_ge_10x_risk": cond_es_ge_k.get(10.0, float("nan")),
            "mc_ttb_ge_5x_risk_p50": ttb_p50.get(5.0, float("nan")),
            "mc_ttb_ge_7x_risk_p50": ttb_p50.get(7.0, float("nan")),
            "mc_ttb_ge_10x_risk_p50": ttb_p50.get(10.0, float("nan")),            "mc_maxdd_xrisk_p50": float(np.percentile(xr, p50)),
            "mc_maxdd_xrisk_p95": float(np.percentile(xr, p95)),
            "mc_maxdd_xrisk_p99": float(np.percentile(xr, p99)),

        })
    # --- Merge policy-aware aggregate into output (just before return) ---
    try:
        if policy_agg is not None:
            out.update({
                "mc_policy": getattr(cfg, "mc_policy", "cap_nav"),
                "mc_cap_nav_ccy": float(getattr(cfg, "mc_cap_nav_ccy", float("nan"))),
                "mc_policy_apply": str(getattr(cfg, "mc_policy_apply", "yearly")),
                "mc_policy_start_month": int(getattr(cfg, "mc_policy_start_month", 1)),
            })
            out.update(policy_agg)
    except Exception:
        pass

    return out



# ----- Column order for monte_carlo_summary.csv (ABSOLUTE first, then NORMALIZED) -----
_MC_COLUMNS_ORDER = [
    "period_start_month",
    "period_end_month",
    "method",
    "block_mean_length_months",
    "n_paths",
    "horizon_months",
    "seed",
    "cagr_annualized_p05",
    "cagr_annualized_p50",
    "cagr_annualized_p95",
    "max_drawdown_magnitude_p50",
    "max_drawdown_magnitude_p95",
    "max_drawdown_magnitude_p99",
    "prob_negative_horizon_return",
    "wealth_multiple_p05",
    "wealth_multiple_p50",
    "wealth_multiple_p95",
    "ending_nav_p05",
    "ending_nav_p50",
    "ending_nav_p95",
    "prob_maxdd_ge_5pc_eom",
    "prob_maxdd_ge_7pc_eom",
    "prob_maxdd_ge_10pc_eom",
    "prob_maxdd_ge_20pc_eom",
    "prob_maxdd_ge_30pc_eom",
    "prob_maxdd_ge_5x_risk_eom",
    "prob_maxdd_ge_7x_risk_eom",
    "prob_maxdd_ge_10x_risk_eom",
    "mc_maxdd_xrisk_p50",
    "mc_maxdd_xrisk_p95",
    "mc_maxdd_xrisk_p99",
    "mc_ttb_ge_5x_risk_p50",
    "mc_ttb_ge_7x_risk_p50",
    "mc_ttb_ge_10x_risk_p50",
    "prob_no_breach_ge_5x_risk_eom",
    "prob_no_breach_ge_7x_risk_eom",
    "prob_no_breach_ge_10x_risk_eom",
    "cond_es_maxdd_ge_5x_risk",
    "cond_es_maxdd_ge_7x_risk",
    "cond_es_maxdd_ge_10x_risk",
    "risk_per_trade_pct",
    'mc_policy',
    'mc_cap_nav_ccy',
    'mc_policy_apply',
    'mc_policy_start_month',
    'policy_total_cash_out_ccy_p05',
    'policy_total_cash_out_ccy_p50',
    'policy_total_cash_out_ccy_p95',
    'policy_ending_nav_ccy_p05',
    'policy_ending_nav_ccy_p50',
    'policy_ending_nav_ccy_p95',
    'policy_flow_aware_ending_value_ccy_p05',
    'policy_flow_aware_ending_value_ccy_p50',
    'policy_flow_aware_ending_value_ccy_p95',
    'policy_flow_aware_multiple_p05',
    'policy_flow_aware_multiple_p50',
    'policy_flow_aware_multiple_p95',
    'policy_prob_any_cashout',
    'policy_avg_cashout_events']

def _write_monte_carlo_summary_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """
    Write Monte Carlo summary CSV with consistent rounding.
    Extended to support policy-aware fields without touching older columns' behavior.
    """
    patterns_decimals = [
        (["*_*ccy*","*_ccy"], 2),
        (["ending_nav_p05","ending_nav_p50","ending_nav_p95"], 2),
        (["ending_nav_*","n_paths","horizon_months","block_mean_length_months","seed","mc_ttb_*"], 0),
        (["wealth_multiple_*","cond_es_maxdd_ge_*x_risk","mc_maxdd_xrisk_*"], 2),
        (["*cagr*","*return*","max_drawdown_magnitude_*","prob_*","risk_per_trade_pct"], 4),
        # Policy-aware rounding rules (money 2dp, multiples 4dp)
        (["policy_*_ccy_*","mc_cap_nav_ccy"], 2),
        (["policy_*_multiple*"], 4),
    
        (["*_*ccy*","*_ccy"], 2),
        (["ending_nav_p05","ending_nav_p50","ending_nav_p95"], 2),]
    df_out = _apply_rounding_by_patterns_isolated(df, patterns_decimals, int_cols=set())
    df_out.to_csv(csv_path, index=False)



# ---------- Schema / rounding loader ----------
def _load_schema_rounding_or_thresholds(schema_path="/mnt/data/advanced_metrics_schema.json"):
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        min_trades = 12
        try:
            pol = data.get("methodology", {}).get("minimum_sample_policy", {})
            if isinstance(pol.get("min_trades_per_year_warning"), int):
                min_trades = pol["min_trades_per_year_warning"]
        except Exception:
            pass
        rounding = data.get("rounding_policy", {})
        return {"min_trades_per_year_warning": min_trades, "rounding_policy": rounding, "raw_schema": data}
    except Exception:
        return {"min_trades_per_year_warning": 12, "rounding_policy": {}, "raw_schema": {}}

_SCHEMA_CACHE = _load_schema_rounding_or_thresholds()

def _schema_headers_for(fname: str):
    data = _SCHEMA_CACHE.get("raw_schema", {})
    for key in ("column_order", "columns", "headers", "expected_headers"):
        node = data.get(key, {})
        if isinstance(node, dict) and isinstance(node.get(fname), list):
            return list(node.get(fname))
    return None



def _warn(msg: str):
    """Lightweight stderr logger used for gentle deprecation and validation messages."""
    try:
        import sys
        sys.stderr.write(str(msg).rstrip() + "\n")
    except Exception:
        pass



# --- Monte Carlo Policy Helpers (Variant A: NAV cap) ---
import numpy as _np  # safe re-import; idempotent if already imported

def _mc_month_index(h: int, start_month: int = 1) -> "_np.ndarray":
    """
    Return a month-of-year index of length h, cycling through 1..12,
    anchored at `start_month`. Example (h=5, start_month=11): [11, 12, 1, 2, 3].
    Used to detect the 'year boundary' step for policy application.
    """
    if not isinstance(h, int) or h <= 0:
        raise ValueError("h must be a positive integer")
    if not (1 <= int(start_month) <= 12):
        raise ValueError("start_month must be in 1..12")
    start = int(start_month)
    return ((start - 1 + _np.arange(h, dtype=int)) % 12) + 1


def _mc_apply_policy_cap_nav(
    path_returns: "_np.ndarray",
    starting_nav: float,
    cap_nav: float,
    apply_mode: str = "yearly",
    start_month: int = 1,
    tolerance: float = 0.0,
) -> dict:
    """
    Apply a NAV cap policy along one simulated path of monthly returns.

    Yearly mode:
      - At 'year-start' steps (month == start_month), BEFORE applying that month's return:
          if NAV > cap + tol: cash_out = NAV - cap; NAV = cap; events += 1.
      - Then apply the monthly return: NAV *= (1 + r_t).

    Monthly mode:
      - AFTER applying each month's return:
          if NAV > cap + tol: cash_out = NAV - cap; NAV = cap; events += 1.

    The function is side-effect free (does not mutate inputs) and runs in O(h).

    Returns:
      {
        "ending_nav_policy": float,
        "total_cash_out": float,
        "num_events": int,
        "flow_aware_ending_value": float,  # ending_nav_policy + total_cash_out
        "flow_aware_multiple": float       # flow_aware_ending_value / starting_nav
      }
    """
    # Basic validation
    try:
        r = _np.asarray(path_returns, dtype=float).copy()
    except Exception as e:
        raise ValueError(f"path_returns must be array-like of floats: {e}")
    if r.ndim != 1 or r.size == 0:
        raise ValueError("path_returns must be a non-empty 1D array")
    if starting_nav <= 0 or cap_nav <= 0:
        raise ValueError("starting_nav and cap_nav must be positive")
    if not (1 <= int(start_month) <= 12):
        raise ValueError("start_month must be in 1..12")
    if apply_mode not in ("yearly", "monthly"):
        raise ValueError("apply_mode must be 'yearly' or 'monthly'")
    tol = float(tolerance) if tolerance is not None else 0.0

    nav = float(starting_nav)
    total_cash_out = 0.0
    events = 0
    h = int(r.size)

    if apply_mode == "yearly":
        months = _mc_month_index(h=h, start_month=int(start_month))
        for t in range(h):
            if months[t] == int(start_month) and (nav > cap_nav + tol):
                cash = nav - cap_nav
                total_cash_out += cash
                nav = cap_nav
                events += 1
            nav *= (1.0 + float(r[t]))
    else:  # monthly
        for t in range(h):
            nav *= (1.0 + float(r[t]))
            if nav > cap_nav + tol:
                cash = nav - cap_nav
                total_cash_out += cash
                nav = cap_nav
                events += 1

    ending_nav = nav
    flow_aware_ending_value = ending_nav + total_cash_out
    flow_aware_multiple = flow_aware_ending_value / float(starting_nav)
    return {
        "ending_nav_policy": float(ending_nav),
        "total_cash_out": float(total_cash_out),
        "num_events": int(events),
        "flow_aware_ending_value": float(flow_aware_ending_value),
        "flow_aware_multiple": float(flow_aware_multiple),
    }



def _mc_aggregate_policy(stats_per_path: "list[dict]") -> dict:
    """
    Aggregate policy-aware statistics across simulated paths.

    Inputs:
      stats_per_path: list of dicts returned by _mc_apply_policy_cap_nav for each path, e.g.:
        {
          "ending_nav_policy": float,
          "total_cash_out": float,
          "num_events": int,
          "flow_aware_ending_value": float,
          "flow_aware_multiple": float
        }

    Returns (raw floats, no rounding):
      - policy_total_cash_out_ccy_p05/p50/p95
      - policy_ending_nav_ccy_p05/p50/p95
      - policy_flow_aware_ending_value_ccy_p05/p50/p95
      - policy_flow_aware_multiple_p05/p50/p95
      - policy_prob_any_cashout      (share of paths with num_events >= 1)
      - policy_avg_cashout_events    (mean num_events across paths)
    """
    if not isinstance(stats_per_path, list) or len(stats_per_path) == 0:
        raise ValueError("stats_per_path must be a non-empty list")

    import numpy as _np

    def _arr(key: str):
        return _np.asarray([float(d.get(key, _np.nan)) for d in stats_per_path], dtype=float)

    tco   = _arr("total_cash_out")
    enav  = _arr("ending_nav_policy")
    fave  = _arr("flow_aware_ending_value")
    fmult = _arr("flow_aware_multiple")
    nevt  = _arr("num_events")

    qs = _np.array([0.05, 0.50, 0.95], dtype=float)

    def _pcts(a):
        # robust to NaNs
        return _np.nanquantile(a, qs, method="linear")

    p_tco   = _pcts(tco)
    p_enav  = _pcts(enav)
    p_fave  = _pcts(fave)
    p_fmult = _pcts(fmult)

    nevt_valid = nevt[~_np.isnan(nevt)]
    prob_any = float(_np.mean(nevt_valid >= 1)) if nevt_valid.size else float("nan")
    avg_evt  = float(_np.mean(nevt_valid)) if nevt_valid.size else float("nan")

    return {
        "policy_total_cash_out_ccy_p05": float(p_tco[0]),
        "policy_total_cash_out_ccy_p50": float(p_tco[1]),
        "policy_total_cash_out_ccy_p95": float(p_tco[2]),

        "policy_ending_nav_ccy_p05": float(p_enav[0]),
        "policy_ending_nav_ccy_p50": float(p_enav[1]),
        "policy_ending_nav_ccy_p95": float(p_enav[2]),

        "policy_flow_aware_ending_value_ccy_p05": float(p_fave[0]),
        "policy_flow_aware_ending_value_ccy_p50": float(p_fave[1]),
        "policy_flow_aware_ending_value_ccy_p95": float(p_fave[2]),

        "policy_flow_aware_multiple_p05": float(p_fmult[0]),
        "policy_flow_aware_multiple_p50": float(p_fmult[1]),
        "policy_flow_aware_multiple_p95": float(p_fmult[2]),

        "policy_prob_any_cashout": prob_any,
        "policy_avg_cashout_events": avg_evt,
    }





# ---------- Config ----------
@dataclass
class Config:
    # --- Monte Carlo policy-aware (Variant A) ---
    # Keep defaults so current behavior remains unchanged.
    mc_policy: str = "none"                 # "none" | "cap_nav"
    mc_cap_nav_ccy: float | None = None       # required when mc_policy="cap_nav"
    mc_policy_apply: str = "yearly"         # "yearly" | "monthly"
    mc_policy_start_month: int = 1           # 1..12 (January by default)
    mc_cap_tolerance_ccy: float = 0.0        # ignore pennies around the cap
    rebase_enable: bool = True  # enable year-boundary rebasing (Dec→Jan)
    rebase_gap_tol_abs_ccy: float = 10.0  # absolute tolerance in account currency
    rebase_write_log: bool = True  # write rebasing_applied.csv audit log

    # ---- Confidence Intervals (CI) ----
    ci_level_pct: int = 90                     # 90% CI => P05/P95
    ci_boot_method: str = "stationary_bootstrap"
    ci_n_boot: int = 5000
    ci_L_months: list[int] = field(default_factory=lambda: [3,6,9,12])  # eom
    ci_L_days:   list[int] = field(default_factory=lambda: [3,5,10])    # intraday
    ci_prob_method: str = "wilson"
    ci_seed: int = 43


    starting_nav: float = 100_000.0
    rf: float = 0.0
    tail_ratio_mode: str = "all"  # V8: preferred base for Tail Ratio ('all' or 'active')                 # monthly risk-free
    stdev_ddof: int = 1
    timezone: str = "UTC"
    risk_per_trade_pct: float = 0.01  # 1%
    nw_lag: int = 6  # used in full-period Newey–West metrics

# ---------- I/O helpers ----------
def discover_years_from_input(input_dir: Path) -> List[int]:
    years = set()
    for pattern in ("trades_*.csv", "trades_*.txt", "balance_*.csv", "balance_*.txt"):
        for p in input_dir.glob(pattern):
            m = re.search(r"(?:trades|balance)_(\d{4})\.", p.name)
            if m:
                years.add(int(m.group(1)))
    return sorted(years)

def find_input_files_for_year(input_dir: Path, year: int) -> Tuple[Optional[Path], Optional[Path]]:
    trades = None
    equity = None
    for ext in ("csv", "txt"):
        t = input_dir / f"trades_{year}.{ext}"
        e = input_dir / f"balance_{year}.{ext}"
        if t.exists():
            trades = t
        if e.exists():
            equity = e
    if trades is None:
        matches = list(input_dir.glob(f"trades_{year}.*"))
        if matches:
            trades = matches[0]
    if equity is None:
        matches = list(input_dir.glob(f"balance_{year}.*"))
        if matches:
            equity = matches[0]
    return trades, equity

def load_equity_minimal(equity_path: Path) -> pd.DataFrame:
    df = pd.read_csv(equity_path, sep=";")
    required = ["time_utc", "capital_ccy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Equity file {equity_path.name} missing columns: {missing}")
    out = df.loc[:, required].copy()
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=False, errors="raise")
    out = out.drop_duplicates(subset=["time_utc"], keep="last").sort_values("time_utc").reset_index(drop=True)
    # NEW (V8): deduplicate identical timestamps; keep the last observation
    out = out.drop_duplicates(subset=["time_utc"], keep="last").sort_values("time_utc").reset_index(drop=True)
    return out

def load_trades_minimal(trades_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_path, sep=";")
    required = ["open_date", "open_time", "close_date", "close_time", "pnl_abs", "pnl_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Trades file {trades_path.name} missing columns: {missing}")
    out = df.loc[:, required].copy()
    out["open_ts_utc"] = out["open_date"].astype(str) + " " + out["open_time"].astype(str)
    out["close_ts_utc"] = out["close_date"].astype(str) + " " + out["close_time"].astype(str)
    out["close_ts_utc"] = pd.to_datetime(out["close_ts_utc"], utc=False, errors="raise")
    return out[["open_ts_utc", "close_ts_utc", "pnl_abs", "pnl_pct"]]

# ---------- Monthly returns ----------
def _eom_nav_from_equity_no_sort(equity_df: pd.DataFrame) -> pd.Series:
    df = equity_df.copy()
    df["__row_idx"] = np.arange(len(df), dtype=np.int64)
    df["__period"] = df["time_utc"].dt.to_period("M")
    max_ts_per = df.groupby("__period")["time_utc"].transform("max")
    df["__row_idx_tie"] = np.where(df["time_utc"].eq(max_ts_per), df["__row_idx"], -1)
    pick_idx = df.groupby("__period")["__row_idx_tie"].idxmax()
    eom = df.loc[pick_idx, ["__period", "capital_ccy"]].set_index("__period")["capital_ccy"].sort_index()
    return eom

def _build_month_grid(eom_nav: pd.Series) -> pd.PeriodIndex:
    start, end = eom_nav.index.min(), eom_nav.index.max()
    return pd.period_range(start, end, freq="M")

def _compute_monthly_returns(eom_nav: pd.Series, starting_nav: float) -> pd.Series:
    grid = _build_month_grid(eom_nav)
    nav = eom_nav.reindex(grid).ffill()
    prev = nav.shift(1)
    if len(nav) > 0:
        prev.iloc[0] = starting_nav
    r_m = nav / prev - 1.0
    r_m.index = grid
    return r_m

def _active_month_flags(trades_df: pd.DataFrame, grid: pd.PeriodIndex) -> pd.Series:
    if trades_df is None or len(trades_df) == 0:
        return pd.Series(False, index=grid)
    per = trades_df["close_ts_utc"].dt.to_period("M")
    counts = per.value_counts().reindex(grid, fill_value=0).sort_index()
    return counts.gt(0)

# ---------- Risk/Return helpers (monthly) ----------
_EPS_DEN = 1e-12
_EPS_ZERO = 1e-12

def _ann_factor() -> float:
    return math.sqrt(12.0)

def _safe_std(x: np.ndarray, ddof: int) -> float:
    if x.size <= ddof:
        return float("nan")
    return float(np.std(x, ddof=ddof))

def _sharpe_annualized_with_rf(monthly_returns: np.ndarray, rf_m: float, ddof: int) -> float:

    sd = _safe_std(monthly_returns, ddof=ddof)
    mu = float(np.mean(monthly_returns - rf_m))
    if not np.isfinite(sd) or abs(sd) < _EPS_DEN:
        if abs(mu) < _EPS_ZERO:
            return float("nan")
        return float("inf") if mu > 0 else float("-inf")
    return (mu / sd) * _ann_factor()

def _downside_std(monthly_returns: np.ndarray, ddof: int, target_m: float = 0.0) -> float:
    neg = monthly_returns[monthly_returns < target_m] - target_m
    if neg.size < 2:
        return float("nan")
    return float(np.std(neg, ddof=ddof))

def _sortino_annualized(monthly_returns: np.ndarray, target_m: float = 0.0, ddof: int = 1) -> tuple[float, bool]:

    x = np.asarray(monthly_returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), True

    mean_excess = float(np.mean(x - target_m))
    neg = x[x < target_m] - target_m

    #  ddof=1
    if neg.size < max(2, ddof + 1):
        return float("nan"), True

    ds = float(np.std(neg, ddof=ddof))

    ratio = _ratio_num_den(mean_excess, ds)  #Inf/NaN
    if isinstance(ratio, float) and not np.isfinite(ratio):
        # ±inf
        return float(ratio), False

    return float(ratio * _ann_factor()), False

# ---------- Newey–West (full-period only) ----------
def _newey_west_long_run_var(x: np.ndarray, q: int) -> float:
    """
    Long-run variance (Bartlett kernel) for monthly returns.
    S = gamma_0 + 2 * sum_{k=1..q} w_k * gamma_k,  w_k = 1 - k/(q+1)
    gamma_k = (1/n) * sum_{t=k..n-1} (x_t - mu)(x_{t-k} - mu)
    """
    if x.size == 0:
        return float("nan")
    n = int(x.size)
    mu = float(np.mean(x))
    e = x - mu
    g0 = float(np.dot(e, e) / n)  # gamma_0
    q = int(max(0, min(q, n - 1)))
    if q == 0:
        return max(g0, 0.0)
    s = g0
    for k in range(1, q + 1):
        w = 1.0 - k / (q + 1.0)
        gk = float(np.dot(e[k:], e[:-k]) / n)
        s += 2.0 * w * gk
    return float(max(s, 0.0))

# ---------- EoM underwater helpers ----------
def _monthly_underwater_metrics_with_trough(r: np.ndarray) -> tuple[float, int, float, int, int, np.ndarray]:
    n = r.size
    if n == 0:
        return float("nan"), 0, float("nan"), -1, 0, np.array([])
    eps = _EPS_DEN
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    maxdd = float(np.min(dd))
    peak_change = np.r_[True, peak[1:] > peak[:-1]]
    peak_index = np.maximum.accumulate(np.where(peak_change, np.arange(n), 0))
    longest_with_recovery = 0
    i = 0
    while i < n:
        if dd[i] < -eps:
            j = i
            while j + 1 < n and dd[j + 1] < -eps:
                j += 1
            recovery_idx = j + 1 if (j + 1 < n and dd[j + 1] >= -eps) else None
            start_peak_i = int(peak_index[i])
            if recovery_idx is not None:
                run_len = int(recovery_idx - start_peak_i + 1)
            else:
                run_len = int(j - start_peak_i + 1)
            if run_len > longest_with_recovery:
                longest_with_recovery = run_len
            i = j + 1
        else:
            i += 1
    trough_idx = int(np.argmin(dd))
    rec_idx = None
    for k in range(trough_idx + 1, n):
        if dd[k] >= -eps:
            rec_idx = k
            break
    ttr = float(rec_idx - trough_idx + 1) if rec_idx is not None else float("nan")
    months_since_trough = int((n - 1) - trough_idx) if trough_idx >= 0 else 0
    return maxdd, int(longest_with_recovery), ttr, trough_idx, months_since_trough, dd

# ---------- RAW equity helpers ----------
def _months_between(a, b) -> int:
    pa = pd.Timestamp(a).to_period("M")
    pb = pd.Timestamp(b).to_period("M")
    return int((pb.year - pa.year) * 12 + (pb.month - pa.month))

def _maxdd_from_intramonth_equity_year(equity_df: pd.DataFrame, year: int) -> float:
    sub = equity_df.loc[equity_df["time_utc"].dt.year == year, ["time_utc", "capital_ccy"]]
    if sub.empty:
        return float("nan")
    sub = sub.sort_values("time_utc", ascending=True, kind="mergesort")
    vals = sub["capital_ccy"].astype(float).to_numpy()
    peaks = np.maximum.accumulate(vals)
    dd = vals / peaks - 1.0
    mdd = float(np.min(dd))
    return mdd if np.isfinite(mdd) else float("nan")

def _maxdd_from_intramonth_equity_full_period(equity_df: pd.DataFrame) -> float:
    if equity_df is None or len(equity_df) == 0:
        return float("nan")
    eq = equity_df[["time_utc", "capital_ccy"]].copy().sort_values("time_utc", ascending=True, kind="mergesort")
    vals = eq["capital_ccy"].astype(float).to_numpy()
    peaks = np.maximum.accumulate(vals)
    dd = vals / peaks - 1.0
    mdd = float(np.min(dd))
    return mdd if np.isfinite(mdd) else float("nan")

def _intramonth_underwater_metrics_months(equity_df: pd.DataFrame) -> tuple[float, float, float]:
    if equity_df is None or len(equity_df) == 0:
        return float("nan"), float("nan"), float("nan")
    eps = _EPS_DEN
    eq = equity_df[["time_utc","capital_ccy"]].copy().sort_values("time_utc", kind="mergesort")
    vals = eq["capital_ccy"].astype(float).to_numpy()
    n = len(vals)
    peaks = np.maximum.accumulate(vals)
    dd = vals / peaks - 1.0
    peak_vals = peaks
    peak_change = np.r_[True, peak_vals[1:] > peak_vals[:-1]]
    peak_index = np.maximum.accumulate(np.where(peak_change, np.arange(n), 0))
    underwater = dd < -eps
    longest_months = 0
    i = 0
    while i < n:
        if underwater[i]:
            j = i
            while j + 1 < n and underwater[j + 1]:
                j += 1
            recovery_idx = j + 1 if (j + 1 < n and dd[j + 1] >= -eps) else None
            start_peak_i = peak_index[i]
            if recovery_idx is not None:
                months_len = _months_between(eq["time_utc"].iloc[start_peak_i], eq["time_utc"].iloc[recovery_idx]) + 1
            else:
                months_len = _months_between(eq["time_utc"].iloc[start_peak_i], eq["time_utc"].iloc[j])
            if months_len > longest_months:
                longest_months = months_len
            i = j + 1
        else:
            i += 1
    trough_i = int(np.argmin(dd))
    rec_i = None
    for k in range(trough_i + 1, n):
        if dd[k] >= -eps:
            rec_i = k
            break
    ttr_months = float("nan") if rec_i is None else float(_months_between(eq["time_utc"].iloc[trough_i], eq["time_utc"].iloc[rec_i]) + 1)
    months_since_trough = float(_months_between(eq["time_utc"].iloc[trough_i], eq["time_utc"].iloc[-1]))
    return float(longest_months), float(ttr_months), months_since_trough


# ---------- Helper: safe ratio num/den ----------
_EPS_DEN  = 1e-12   
_EPS_ZERO = 1e-12   

def _ratio_num_den(num: float, den: float) -> float:
    """generic_ratio:  |den|<eps: +Inf,  num>0; -Inf,  num<0; NaN,  |num|<eps."""
    if num is None or den is None or np.isnan(num) or np.isnan(den):
        return float("nan")
    if abs(den) < _EPS_DEN:
        if abs(num) < _EPS_ZERO:
            return float("nan")
        return float("inf") if num > 0 else float("-inf")
    return num / den

# ---------- Helper: Calmar ratio ----------
def _calmar(ann_return: float, maxdd: float) -> float:
    """
    Calmar = ann_return / |maxdd|.
    Epsilon/Inf - generic_ratio:
      - if |den|<eps: +Inf, if num>0; -Inf, if num<0; NaN, if |num|<eps.
    """
    if maxdd is None or np.isnan(maxdd) or np.isnan(ann_return):
        return float("nan")
    return _ratio_num_den(ann_return, abs(maxdd))




# Max consecutive up/down runs on monthly returns
def _max_consecutive_runs(r: np.ndarray, mode: str) -> int:
    arr = np.asarray(r, dtype=float)
    if mode == 'up':
        cond = arr > 0
    elif mode == 'down':
        cond = arr < 0
    else:
        raise ValueError("mode must be 'up' or 'down'")
    max_run = 0
    cur = 0
    for v in cond:
        if bool(v):
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return int(max_run)


# ---------- DD Quantiles (Full-period EoM) ----------
def _q_linear(x: np.ndarray, q: float) -> float:
    try:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan")
        return float(np.quantile(x, q, method="linear"))
    except Exception:
        return float("nan")

def _dd_quantiles_full_period(monthly_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config) -> dict:
    # Build monthly returns array
    r = monthly_df["monthly_return"].astype(float).to_numpy()
    # Use existing monthly underwater helper to get dd series (EoM-based)
    _, _, _, _, _, dd = _monthly_underwater_metrics_with_trough(r)
    eps = _EPS_DEN
    mask = dd < -eps
    dd_neg = dd[mask]
    dd_obs_cnt = int(mask.sum())

    # Episodes count and durations (contiguous runs of mask==True, in months)
    episodes = 0
    durations = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            episodes += 1
            durations.append(int(j - i + 1))
            i = j + 1
        else:
            i += 1

    # Post-process: keep only CLOSED episodes (drop unfinished trailing episode)
    try:
        if isinstance(mask, np.ndarray) and mask.size > 0 and bool(mask[-1]):
            if isinstance(durations, list) and len(durations) > 0:
                durations = durations[:-1]
            if 'episodes' in locals() and episodes > 0:
                episodes -= 1
    except Exception:
        pass

    # Quantiles of dd depth (negative values)
    p90 = _q_linear(dd_neg, 0.90)
    p95 = _q_linear(dd_neg, 0.95)
    p99 = _q_linear(dd_neg, 0.99)

    # Quantiles of underwater episode durations (months)
    dur_arr = np.asarray(durations, dtype=float) if durations else np.asarray([], dtype=float)
    dur_p90 = _q_linear(dur_arr, 0.90)
    dur_p95 = _q_linear(dur_arr, 0.95)

    # xRisk conversions (R = DD / risk_per_trade_pct)
    risk = float(cfg.risk_per_trade_pct)
    def xr(v):
        if not np.isfinite(risk) or risk <= 0.0 or not np.isfinite(v):
            return float("nan")
        return float(abs(v) / risk)

    row = {
        "period_start_month": (lambda _s: f"{_s.year:04d}-{_s.month:02d}")(pd.PeriodIndex(year=monthly_df["year"].astype(int), month=monthly_df["month"].astype(int), freq="M").min()),
        "period_end_month": (lambda _e: f"{_e.year:04d}-{_e.month:02d}")(pd.PeriodIndex(year=monthly_df["year"].astype(int), month=monthly_df["month"].astype(int), freq="M").max()),
        "dd_observations_count": dd_obs_cnt,
        "dd_episodes_count": episodes,
        "drawdown_p90_full_period": p90,
        "drawdown_p95_full_period": p95,
        "drawdown_p99_full_period": p99,
        "underwater_duration_p90_full_period": dur_p90,
        "underwater_duration_p95_full_period": dur_p95,
        "drawdown_p90_full_period_xrisk": xr(p90),
        "drawdown_p95_full_period_xrisk": xr(p95),
        "drawdown_p99_full_period_xrisk": xr(p99),
    }
    return row

def _format_dd_quantiles_for_csv(row: dict) -> pd.DataFrame:
    # Column order mirrors the schema, but is hardcoded to avoid runtime dependency.
    cols = [
        "period_start_month",
        "period_end_month",
        "dd_observations_count",
        "dd_episodes_count",
        "drawdown_p90_full_period",
        "drawdown_p95_full_period",
        "drawdown_p99_full_period",
        "underwater_duration_p90_full_period",
        "underwater_duration_p95_full_period",
        "drawdown_p90_full_period_xrisk",
        "drawdown_p95_full_period_xrisk",
        "drawdown_p99_full_period_xrisk",
    ]
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in cols}])
    return df

# Hardcoded rounding for dd_quantiles_full_period.csv (schema-independent)
def _decimals_dd_quantiles(col: str) -> int | None:
    n = str(col)
    if n.endswith("_xrisk"):
        return 2
    if "drawdown_" in n:
        return 4
    if "underwater_duration_" in n:
        return 0
    if n in ("dd_observations_count", "dd_episodes_count"):
        return 0
    if n in ("period_start_month", "period_end_month"):
        return None
    return None

def _apply_hardcoded_rounding_dd_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        d = _decimals_dd_quantiles(c)
        if d is not None:
            out[c] = out[c].map(lambda x: _fmt_value_for_csv(x, d))
    return out

def write_dd_quantiles_full_period_csv(monthly_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config, csv_path: Path) -> None:
    row = _dd_quantiles_full_period(monthly_df, equity_df, cfg)
    df_out = _format_dd_quantiles_for_csv(row)
    df_out = _apply_hardcoded_rounding_dd_quantiles(df_out)
    df_out.to_csv(csv_path, index=False)

# ---------- Rolling 12-month window metrics ----------
def _quantile_linear(x: np.ndarray, q: float) -> float:
    # Reuse a safe quantile with linear method (type 7); independent of JSON
    try:
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.quantile(arr, q, method="linear"))
    except Exception:
        return float("nan")

def _maxdd_eom_from_returns(r: np.ndarray) -> float:
    # Compute EoM max drawdown from a returns vector by compounding to NAV=1 at start
    try:
        r = np.asarray(r, dtype=float)
        r = r[np.isfinite(r)]
        if r.size == 0:
            return float("nan")
        nav = np.cumprod(1.0 + r, dtype=float)
        peak = np.maximum.accumulate(nav)
        dd = (nav / peak) - 1.0
        return float(np.min(dd)) if dd.size else float("nan")
    except Exception:
        return float("nan")

def _rolling_12m_metrics(monthly_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Prepare monthly PeriodIndex for robust window bounds
    periods = pd.PeriodIndex(year=monthly_df["year"].astype(int),
                             month=monthly_df["month"].astype(int), freq="M")
    order = np.argsort(periods.values)
    r = monthly_df["monthly_return"].to_numpy(dtype=float)[order]
    act = monthly_df["active_month"].to_numpy(dtype=bool)[order] if "active_month" in monthly_df.columns else np.ones_like(r, dtype=bool)
    periods = periods[order]

    rows = []
    win = 12
    n = len(r)
    for end_idx in range(win - 1, n):
        start_idx = end_idx - (win - 1)
        r_win = r[start_idx:end_idx+1]
        p_win = periods[start_idx:end_idx+1]
        a_win = act[start_idx:end_idx+1]

        insufficient = bool(np.sum(np.isfinite(r_win)) < win)
        # Default row structure
        start_p = p_win[0]
        end_p = p_win[-1]
        row = {
            "month": f"{end_p.year:04d}-{end_p.month:02d}",
            "window_months": win,
            "window_start_month": f"{start_p.year:04d}-{start_p.month:02d}",
            "window_end_month": f"{end_p.year:04d}-{end_p.month:02d}",
            "insufficient_months": False,  # by construction: only full 12m windows emitted
        }

        if insufficient:
            # Skip windows with <12 valid months entirely (per methodology)
            continue

        # All metrics for full 12 observations
        r_finite = r_win[np.isfinite(r_win)]
        # 12M return
        roll_ret = float(np.prod(1.0 + r_finite, dtype=float) - 1.0) if r_finite.size == win else float("nan")

        # Vol (annualized)
        sd = _safe_std(r_win, ddof=cfg.stdev_ddof)
        vol_ann = float(sd * _ann_factor())

        # Sharpe (annualized) with monthly RF
        sharpe = _sharpe_annualized_with_rf(r_win, cfg.rf, ddof=cfg.stdev_ddof)

        # Sortino (annualized), track insufficient_negative_months
        sortino, insuff_negs = _sortino_annualized(r_win, ddof=cfg.stdev_ddof)

        # MaxDD in window (EoM)
        mdd = _maxdd_eom_from_returns(r_win)

        # Calmar
        calmar = _calmar(roll_ret, mdd)

        # Activity / sign counts
        active_cnt = int(np.sum(a_win))
        pos_cnt = int(int(_classify_sign_masks(r_win)[0].sum()))
        neg_cnt = int(int(_classify_sign_masks(r_win)[1].sum()))

        # Omega (target 0m)
        pos_sum = float(np.sum(np.clip(r_win, 0.0, None)))
        neg_sum = float(np.sum(np.clip(r_win, None, 0.0)))
        neg_sum_abs = abs(neg_sum)
        omega = (pos_sum / neg_sum_abs) if neg_sum_abs > 0.0 else float("nan")

        # VaR/ES 95
        q05 = _quantile_linear(r_finite, 0.05)
        es95 = float(np.mean(r_finite[r_finite <= q05])) if np.isfinite(q05) and np.any(r_finite <= q05) else float("nan")

        # Tail ratio p95/p5
        q95 = _quantile_linear(r_finite, 0.95)
        tail_ratio = (q95 / abs(q05)) if (np.isfinite(q95) and np.isfinite(q05) and abs(q05) > 0.0) else float("nan")

        # xRisk scaling (divide by risk per trade in decimals)
        risk = float(cfg.risk_per_trade_pct)
        def xr(v: float) -> float:
            return float(abs(v) / risk) if (np.isfinite(v) and risk > 0.0) else float("nan")

        row.update({
            "rolling_return_12m": roll_ret,
            "rolling_volatility_annualized_12m": vol_ann,
            "rolling_sharpe_annualized_12m": sharpe,
            "rolling_sortino_annualized_12m": sortino,
            "rolling_max_drawdown_12m": mdd,
            "rolling_calmar_12m": calmar,
            "active_months_count_12m": active_cnt,
            "active_months_share_12m": float(active_cnt / float(win)),
            "positive_months_count_12m": pos_cnt,
            "negative_months_count_12m": neg_cnt,
            "insufficient_negative_months": insuff_negs,
            "positive_months_share_12m": float(pos_cnt / float(win)),
            "negative_months_share_12m": float(neg_cnt / float(win)),
            "rolling_omega_12m_target_0m": omega,
            "rolling_var_95_12m": q05,
            "rolling_es_95_12m": es95,
            "rolling_tail_ratio_p95_p5_12m": tail_ratio,
            "rolling_max_drawdown_12m_xrisk": xr(mdd),
            "rolling_var_95_12m_xrisk": xr(q05),
            "rolling_es_95_12m_xrisk": xr(es95),
        })
        rows.append(row)

    # Build DataFrame ordered by month
    out = pd.DataFrame(rows)
    # Ensure chronological order
    if not out.empty:
        pi = pd.PeriodIndex(out["month"], freq="M")
        out = out.iloc[np.argsort(pi.values)]
    return out

def _format_rolling_12m_for_csv(df_roll: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "month",
        "window_months",
        "window_start_month",
        "window_end_month",
        "rolling_return_12m",
        "rolling_volatility_annualized_12m",
        "rolling_sharpe_annualized_12m",
        "insufficient_months",
        "rolling_max_drawdown_12m",
        "rolling_calmar_12m",
        "rolling_sortino_annualized_12m",
        "active_months_count_12m",
        "active_months_share_12m",
        "positive_months_count_12m",
        "negative_months_count_12m",
        "insufficient_negative_months",
        "positive_months_share_12m",
        "negative_months_share_12m",
        "rolling_omega_12m_target_0m",
        "rolling_var_95_12m",
        "rolling_es_95_12m",
        "rolling_tail_ratio_p95_p5_12m",
        "rolling_max_drawdown_12m_xrisk",
        "rolling_var_95_12m_xrisk",
        "rolling_es_95_12m_xrisk",
    ]
    if df_roll is None or df_roll.empty:
        return pd.DataFrame(columns=cols)
    # enforce column order
    df = pd.DataFrame({c: df_roll.get(c, np.nan) for c in cols})
    return df


# =========================================================
# ROLLING 36M
# =========================================================

def _rolling_36m_metrics(monthly_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Compute 36-month rolling metrics. Emit rows only for full 36-month windows.
    """
    # Prepare arrays
    yr = monthly_df["year"].to_numpy(int)
    mo = monthly_df["month"].to_numpy(int)
    r  = monthly_df["monthly_return"].to_numpy(float)
    a  = monthly_df["active_month"].to_numpy(bool) if "active_month" in monthly_df.columns else np.ones_like(r, dtype=bool)

    n = len(r); win = 36
    rows = []
    for end in range(win - 1, n):
        start = end - (win - 1)
        r_win = r[start:end+1]
        a_win = a[start:end+1]
        if np.sum(np.isfinite(r_win)) < win:
            # skip insufficient windows
            continue

        # Calendar labels
        end_period = pd.Period(year=int(yr[end]), month=int(mo[end]), freq="M")
        start_period = pd.Period(year=int(yr[start]), month=int(mo[start]), freq="M")

        # Core metrics
        prod = float(np.prod(1.0 + r_win)) - 1.0
        # Annualized return over 36 months (CAGR)
        try:
            ret_ann = float((1.0 + prod)**(12.0/36.0) - 1.0)
        except Exception:
            ret_ann = float("nan")

        sd = float(np.std(r_win, ddof=cfg.stdev_ddof))
        vol_ann = float(sd * np.sqrt(12.0))
        shp     = _sharpe_annualized_with_rf(r_win, cfg.rf, ddof=cfg.stdev_ddof)
        srt, insuff_negs = _sortino_annualized(r_win, ddof=cfg.stdev_ddof)
        mdd = _maxdd_eom_from_returns(r_win)
        cal = _calmar(ret_ann, mdd)

        # Signs / activity
        pos_cnt = int(int(_classify_sign_masks(r_win)[0].sum()))
        neg_cnt = int(int(_classify_sign_masks(r_win)[1].sum()))
        act_cnt = int(np.sum(a_win))

        pos_shr = float(pos_cnt / float(win))
        neg_shr = float(neg_cnt / float(win))
        act_shr = float(act_cnt / float(win))

        # Tails
        q05 = float(np.quantile(r_win, 0.05, method='linear'))
        q95 = float(np.quantile(r_win, 0.95, method='linear'))
        es  = float(np.mean(r_win[r_win <= q05])) if np.any(r_win <= q05) else float('nan')
        tail_ratio = float(q95/abs(q05)) if (np.isfinite(q95) and np.isfinite(q05) and abs(q05) > 0) else float("nan")

        # Omega (target 0m)
        pos_sum = float(np.sum(np.clip(r_win, 0.0, None)))
        neg_sum = float(np.sum(np.clip(r_win, None, 0.0)))
        omega = (pos_sum / abs(neg_sum)) if abs(neg_sum) > 0.0 else float('nan')

        # xRisk scaling
        risk = float(cfg.risk_per_trade_pct) if float(cfg.risk_per_trade_pct) > 0 else float("nan")
        mdd_x = (mdd / risk) if (np.isfinite(risk) and abs(risk) > 0.0) else float('nan')
        var_x = (q05 / risk) if (np.isfinite(risk) and abs(risk) > 0.0) else float('nan')
        es_x  = (es  / risk) if (np.isfinite(risk) and abs(risk) > 0.0) else float('nan')

        rows.append({
            "year": int(yr[end]),
            "month": f"{end_period.year:04d}-{end_period.month:02d}",
            "window_months": int(win),
            "window_start_month": f"{start_period.year:04d}-{start_period.month:02d}",
            "window_end_month": f"{end_period.year:04d}-{end_period.month:02d}",
            "rolling_return_annualized_36m": ret_ann,
            "rolling_max_drawdown_36m": mdd,
            "rolling_calmar_36m": cal,
            "insufficient_months": False,
            "rolling_volatility_annualized_36m": vol_ann,
            "rolling_sharpe_annualized_36m": shp,
            "rolling_sortino_annualized_36m": srt,
            "active_months_count_36m": act_cnt,
            "active_months_share_36m": act_shr,
            "positive_months_count_36m": pos_cnt,
            "negative_months_count_36m": neg_cnt,
            "insufficient_negative_months": bool(insuff_negs),
            "positive_months_share_36m": pos_shr,
            "negative_months_share_36m": neg_shr,
            "rolling_omega_36m_target_0m": omega,
            "rolling_var_95_36m": q05,
            "rolling_es_95_36m": es,
            "rolling_tail_ratio_p95_p5_36m": tail_ratio,
            "rolling_max_drawdown_36m_xrisk": (abs(mdd) / risk) if (np.isfinite(risk) and abs(risk) > 0.0 and np.isfinite(mdd)) else float("nan"),
            "rolling_var_95_36m_xrisk": (abs(q05) / risk) if (np.isfinite(risk) and abs(risk) > 0.0 and np.isfinite(q05)) else float("nan"),
            "rolling_es_95_36m_xrisk": (abs(es) / risk) if (np.isfinite(risk) and abs(risk) > 0.0 and np.isfinite(es)) else float("nan"),
        })

    
    # --- Final sign normalization for drawdowns (negative decimals) ---
    try:
        df_tmp = pd.DataFrame(rows)
        dd_keys = {
            "eom_max_drawdown_full_period",
            "eom_max_drawdown_intra_year",
            "intramonth_max_drawdown_full_period",
            "intramonth_max_drawdown_intra_year",
        }
        if not df_tmp.empty:
            msk = df_tmp["metric_key"].isin(dd_keys)
            if msk.any():
                for col in ["estimate","ci_low","ci_high"]:
                    df_tmp.loc[msk, col] = -df_tmp.loc[msk, col].abs()
                # ensure low <= high
                low = df_tmp.loc[msk, "ci_low"].to_numpy()
                high = df_tmp.loc[msk, "ci_high"].to_numpy()
                new_low = np.minimum(low, high)
                new_high = np.maximum(low, high)
                df_tmp.loc[msk, "ci_low"] = new_low
                df_tmp.loc[msk, "ci_high"] = new_high
        return df_tmp
    except Exception:
        return pd.DataFrame(rows)



def _format_rolling_36m_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "month",
        "window_months",
        "window_start_month",
        "window_end_month",
        "rolling_return_annualized_36m",
        "rolling_max_drawdown_36m",
        "rolling_calmar_36m",
        "insufficient_months",
        "rolling_volatility_annualized_36m",
        "rolling_sharpe_annualized_36m",
        "rolling_sortino_annualized_36m",
        "active_months_count_36m",
        "active_months_share_36m",
        "positive_months_count_36m",
        "negative_months_count_36m",
        "insufficient_negative_months",
        "positive_months_share_36m",
        "negative_months_share_36m",
        "rolling_omega_36m_target_0m",
        "rolling_var_95_36m",
        "rolling_es_95_36m",
        "rolling_tail_ratio_p95_p5_36m",
        "rolling_max_drawdown_36m_xrisk",
        "rolling_var_95_36m_xrisk",
        "rolling_es_95_36m_xrisk",
    ]
    out = df.copy()
    out["month"] = out["month"].astype(str)
    return out[cols]


def _decimals_rolling_36m(col: str):
    name = col
    if name.endswith("_xrisk"):
        return 2
    if ("sharpe" in name) or ("sortino" in name) or ("calmar" in name):
        return 2
    if ("omega" in name) or ("tail_ratio" in name):
        return 2
    if ("return" in name) or ("share" in name) or ("drawdown" in name) or ("volatility_" in name) or ("var_" in name) or ("_es_" in name):
        return 4
    if name in ("window_months", "active_months_count_36m", "positive_months_count_36m", "negative_months_count_36m"):
        return 0
    return None


def _apply_hardcoded_rounding_rolling_36m(df: pd.DataFrame) -> pd.DataFrame:
    def fmt(x, n):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "inf" if (isinstance(x, float) and np.isinf(x)) else "NaN"
        try:
            getcontext().prec = 28
            d = Decimal(str(x))
            q = Decimal("1") if n == 0 else Decimal("1").scaleb(-n)
            s = str(d.quantize(q, rounding=ROUND_HALF_UP))
            if s == "-0" or s == "-0.0" or s == "-0.00" or s == "-0.000" or s == "-0.0000":
                s = s.replace("-", "")
            return s
        except Exception:
            return str(x)

    out = df.copy()
    for c in out.columns:
        n = _decimals_rolling_36m(c)
        if n is None:
            continue
        out[c] = [fmt(v, n) if pd.notna(v) else "NaN" for v in out[c].tolist()]
    return out


def write_rolling_36m_csv(df_roll: pd.DataFrame, csv_path: Path) -> None:
    df_out = _format_rolling_36m_for_csv(df_roll)
    df_out = _apply_hardcoded_rounding_rolling_36m(df_out)
    df_out.to_csv(csv_path, index=False)


def _decimals_rolling_12m(col: str) -> int | None:
    name = str(col)
    if name.endswith("_xrisk"):
        return 2
    if ("sharpe" in name) or ("sortino" in name) or ("calmar" in name):
        return 2
    if ("omega" in name) or ("tail_ratio" in name):
        return 2
    if ("return" in name) or ("share" in name) or ("drawdown" in name) or ("volatility_" in name) \
       or ("var_" in name) or ("_es_" in name):
        return 4
    if name in ("window_months", "active_months_count_12m", "positive_months_count_12m", "negative_months_count_12m"):
        return 0
    # Booleans and dates -> None (no rounding)
    return None

def _apply_hardcoded_rounding_rolling_12m(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        d = _decimals_rolling_12m(c)
        if d is not None:
            out[c] = out[c].map(lambda x: _fmt_value_for_csv(x, d))
    return out

def write_rolling_12m_csv(df_roll: pd.DataFrame, csv_path: Path) -> None:
    df_out = _format_rolling_12m_for_csv(df_roll)
    df_out = _apply_hardcoded_rounding_rolling_12m(df_out)
    df_out.to_csv(csv_path, index=False)
# ---------- Core helpers ----------

# ---------- Hardcoded CSV rounding for FULL-PERIOD (schema-independent) ----------


def _decimals_full_period_hardcoded(col_name: str) -> int | None:
    name = str(col_name)
    # Priority order matters (specific before generic):
    # 1) xRisk & p-values
    if name.endswith("_xrisk"):
        return 2
    if "p_value" in name:
        return 3
    # 2) fixed-named fields
    if name == "ending_nav_full_period":
        return 0
    if name == "wealth_multiple_full_period":
        return 2
    # 3) Sharpe/Sortino/Calmar family
    if ("sharpe" in name) or ("sortino" in name) or ("calmar" in name):
        return 2
    # 4) Ratio-like family
    if ("martin" in name) or ("pain_ratio" in name) or ("gain_to_pain" in name) or ("omega" in name) or ("tail_ratio" in name) or ("skew" in name) or ("kurt" in name):
        return 2
    # 5) Percent-like family (4 decimals)
    if ("cagr" in name) or ("return" in name) or ("volatility" in name) or ("drawdown" in name) or ("share" in name) or name.startswith("annual_return_") or name.startswith("prob_") or ("var_" in name) or ("_es_" in name):
        return 4
    # counts/strings/dates: None (no rounding)
    return None


def _apply_hardcoded_rounding_full(df_out: pd.DataFrame) -> pd.DataFrame:
    if df_out is None or df_out.empty:
        return df_out
    out = df_out.copy()
    for c in list(out.columns):
        dec = _decimals_full_period_hardcoded(str(c))
        if dec is not None:
            out[c] = out[c].map(lambda x: _fmt_value_for_csv(x, dec))
    return out



# ---------- CSV Rounding helpers (used only for full-period writer) ----------
def _wildcard_to_regex(pat: str) -> str:
    pat = re.escape(pat).replace(r"\*", ".*")
    return "^" + pat + "$"

def _decimals_for_column_by_policy(col_name: str, rounding_policy: dict) -> int|None:
    if not isinstance(rounding_policy, dict):
        return None
    priority = rounding_policy.get("priority_order", [])
    for key in priority:
        cfg = rounding_policy.get(key, {})
        exact = cfg.get("applies_to", []) or []
        pats  = cfg.get("applies_to_patterns", []) or []
        dec   = cfg.get("decimals", None)
        if col_name in exact and dec is not None:
            return int(dec)
        for p in pats:
            if re.match(_wildcard_to_regex(p), col_name):
                return int(dec) if dec is not None else None
    return None

def _apply_rounding_by_policy_for_full(df_out: pd.DataFrame, rounding_policy: dict) -> pd.DataFrame:
    # Apply only categories defined in schema.rounding_policy (global),
    # ignoring per-file (which may be empty). Uses existing _fmt_value_for_csv.
    if not isinstance(rounding_policy, dict) or df_out is None or df_out.empty:
        return df_out
    out = df_out.copy()
    for c in list(out.columns):
        d = _decimals_for_column_by_policy(str(c), rounding_policy)
        if d is not None:
            out[c] = out[c].map(lambda x: _fmt_value_for_csv(x, d))
    return out




# ---------- Trades: full-period metrics (clean implementation) ----------

def _tr_build_R_and_time(trades_df: pd.DataFrame, risk_per_trade_pct: float, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return R array and open/close timestamps (np.datetime64[ns]) for closed trades. R = (pnl_pct/100) / risk."""
    if trades_df is None or len(trades_df) == 0 or risk_per_trade_pct is None or risk_per_trade_pct <= 0:
        return np.array([], dtype=float), np.array([], dtype='datetime64[ns]'), np.array([], dtype='datetime64[ns]')
    df = trades_df.copy()
    # Ensure datetime
    if "open_ts_utc" in df.columns:
        try:
            df["open_ts_utc"] = pd.to_datetime(df["open_ts_utc"], utc=False, errors="coerce")
        except Exception:
            df["open_ts_utc"] = pd.NaT
    elif {"open_date","open_time"}.issubset(df.columns):
        df["open_ts_utc"] = pd.to_datetime(df["open_date"].astype(str) + " " + df["open_time"].astype(str), utc=False, errors="coerce")
    if "close_ts_utc" in df.columns:
        try:
            df["close_ts_utc"] = pd.to_datetime(df["close_ts_utc"], utc=False, errors="coerce")
        except Exception:
            df["close_ts_utc"] = pd.NaT
    elif {"close_date","close_time"}.issubset(df.columns):
        df["close_ts_utc"] = pd.to_datetime(df["close_date"].astype(str) + " " + df["close_time"].astype(str), utc=False, errors="coerce")
    # Closed only
    df = df.dropna(subset=["close_ts_utc"])
    # R from pnl_pct
    R = (df["pnl_pct"].astype(float) / 100.0) / float(risk_per_trade_pct)
    return R.to_numpy(dtype=float), df["open_ts_utc"].to_numpy(dtype="datetime64[ns]"), df["close_ts_utc"].to_numpy(dtype="datetime64[ns]")


def _tr_basic_stats(R: np.ndarray, eps: float) -> dict:
    n = int(len(R))
    if n == 0:
        return {
            "trade_count": 0, "win_rate": float("nan"), "profit_factor": float("nan"),
            "average_winning_trade_r": float("nan"), "average_losing_trade_r": float("nan"),
            "payoff_ratio": float("nan"), "expectancy_mean_r": float("nan"), "expectancy_median_r": float("nan"),
            "r_std_dev": float("nan"), "r_min": float("nan"), "r_max": float("nan")
        }
    nz = R[np.abs(R) > eps]
    if len(nz) == 0:
        win_rate = float("nan"); pos_sum = 0.0; neg_sum = 0.0
        avg_win = float("nan"); avg_loss = float("nan"); payoff = float("nan")
    else:
        win_rate = float(_classify_sign_masks(nz)[0].sum() / float(len(nz)))
        pos = nz[nz > 0]; neg = nz[nz < 0]
        pos_sum = float(pos.sum()) if len(pos) > 0 else 0.0
        neg_sum = float(-neg.sum()) if len(neg) > 0 else 0.0
        avg_win = float(np.mean(pos)) if len(pos) > 0 else float("nan")
        avg_loss = float(np.mean(neg)) if len(neg) > 0 else float("nan")
        payoff = float(avg_win / abs(avg_loss)) if (np.isfinite(avg_win) and np.isfinite(avg_loss) and abs(avg_loss) > 0) else float("nan")
    exp_mean = float(np.mean(R)) if n > 0 else float("nan")
    exp_median = float(np.median(R)) if n > 0 else float("nan")
    r_std = float(np.std(R, ddof=1)) if n > 1 else float("nan")
    r_min = float(np.min(R)) if n > 0 else float("nan")
    r_max = float(np.max(R)) if n > 0 else float("nan")
    pf = float(pos_sum / neg_sum) if neg_sum > 0 else (float("inf") if pos_sum > 0 else float("nan"))
    return {
        "trade_count": n, "win_rate": win_rate, "profit_factor": pf,
        "average_winning_trade_r": avg_win, "average_losing_trade_r": avg_loss, "payoff_ratio": payoff,
        "expectancy_mean_r": exp_mean, "expectancy_median_r": exp_median, "r_std_dev": r_std, "r_min": r_min, "r_max": r_max
    }


def _tr_worst_k_run(R: np.ndarray, k: int) -> float:
    n = int(len(R))
    if n < k or k <= 0:
        return float("nan")
    c = np.concatenate(([0.0], np.cumsum(R)))
    # min sum over length-k windows: min_j (c[j] - c[j-k])
    w = c[k:] - c[:-k]
    m = float(np.min(w)) if len(w) > 0 else float("nan")
    # If minimum positive (no loss streak), report 0.0 per convention (drawdown is non-positive)
    return m if (np.isfinite(m) and m <= 0.0) else 0.0


def _tr_cum_maxdd(vals: np.ndarray) -> float:
    """Max drawdown of a cumulative path (vals are cumulative R). Return non-positive value (0 if never down)."""
    if vals.size == 0:
        return float("nan")
    peak = -1e100
    mdd = 0.0
    for x in vals:
        if x > peak:
            peak = x
        dd = x - peak
        if dd < mdd:
            mdd = dd
    return float(mdd)


def _tr_stationary_bootstrap_indices(n: int, horizon: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Return indices for a stationary bootstrap path of given horizon from 0..n-1 with block prob p (mean length 1/p)."""
    idx = np.empty(horizon, dtype=int)
    if n <= 0:
        return idx
    # Start at a random point
    idx[0] = int(rng.integers(0, n))
    for t in range(1, horizon):
        if rng.random() < p:
            # New block: jump anywhere
            idx[t] = int(rng.integers(0, n))
        else:
            # Continue block
            idx[t] = (idx[t-1] + 1) % n
    return idx


def _tr_max_losing_streak(path_R: np.ndarray) -> int:
    best = 0; cur = 0
    for x in path_R:
        if x < 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _tr_bootstrap_metrics(R: np.ndarray, n_paths: int = 5000, horizon: int = 100, p: float = 0.2, rng_seed: int = 12345) -> dict:
    """Compute EDR and streak metrics via stationary bootstrap. Returns dict with required fields; NaNs if no trades."""
    if R is None or len(R) == 0:
        return {
            "edr_100_trades_p50_r": float("nan"),
            "edr_100_trades_p95_r": float("nan"),
            "prob_maxdd_100trades_le_5r": float("nan"),
            "prob_maxdd_100trades_le_7r": float("nan"),
            "prob_maxdd_100trades_le_10r": float("nan"),
            "losing_streak_max_p50_100trades": float("nan"),
            "losing_streak_max_p95_100trades": float("nan"),
            "prob_losing_streak_ge_7_100trades": float("nan"),
            "prob_losing_streak_ge_10_100trades": float("nan"),
        }
    n = len(R)
    rng = np.random.default_rng(rng_seed)
    maxdds = np.empty(n_paths, dtype=float)
    streaks = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        idx = _tr_stationary_bootstrap_indices(n, horizon, p, rng)
        path = R[idx]
        cum = np.cumsum(path)
        maxdds[i] = _tr_cum_maxdd(cum)  # <= 0.0
        streaks[i] = _tr_max_losing_streak(path)
    # EDR percentiles (should be non-positive)
    edr_p50 = float(np.quantile(maxdds, 0.50, method='linear'))
    edr_p95 = float(np.quantile(maxdds, 0.95, method='linear'))
    # Probabilities for thresholds -5R, -7R, -10R (<= threshold)
    p_le_5  = float(np.mean(maxdds <= -5.0))
    p_le_7  = float(np.mean(maxdds <= -7.0))
    p_le_10 = float(np.mean(maxdds <= -10.0))
    # Streak percentiles and probabilities
    st_p50 = float(np.quantile(streaks, 0.50, method='linear'))
    st_p95 = float(np.quantile(streaks, 0.95, method='linear'))
    p_ge_7  = float(np.mean(streaks >= 7.0))
    p_ge_10 = float(np.mean(streaks >= 10.0))
    return {
        "edr_100_trades_p50_r": edr_p50,
        "edr_100_trades_p95_r": edr_p95,
        "prob_maxdd_100trades_le_5r": p_le_5,
        "prob_maxdd_100trades_le_7r": p_le_7,
        "prob_maxdd_100trades_le_10r": p_le_10,
        "losing_streak_max_p50_100trades": st_p50,
        "losing_streak_max_p95_100trades": st_p95,
        "prob_losing_streak_ge_7_100trades": p_ge_7,
        "prob_losing_streak_ge_10_100trades": p_ge_10,
    }


def _tr_holding_period_minutes(trades_df: pd.DataFrame, R: np.ndarray, eps: float) -> dict:
    if trades_df is None or len(trades_df) == 0 or R is None or len(R) == 0:
        return {
            "holding_period_mean_minutes": float("nan"),
            "holding_period_median_minutes": float("nan"),
            "holding_period_p95_minutes": float("nan"),
            "holding_period_mean_minutes_wins": float("nan"),
            "holding_period_mean_minutes_losses": float("nan"),
        }
    df = trades_df.copy()
    try:
        df["open_ts_utc"] = pd.to_datetime(df["open_ts_utc"], utc=False, errors="coerce")
    except Exception:
        pass
    try:
        df["close_ts_utc"] = pd.to_datetime(df["close_ts_utc"], utc=False, errors="coerce")
    except Exception:
        pass
    df = df.dropna(subset=["open_ts_utc","close_ts_utc"]).reset_index(drop=True)
    dur_min = (df["close_ts_utc"] - df["open_ts_utc"]).dt.total_seconds().to_numpy() / 60.0
    # Align length with R (both after dropping NaTs above we recompute R if lengths mismatch)
    n = min(len(R), len(dur_min))
    if n == 0:
        return {
            "holding_period_mean_minutes": float("nan"),
            "holding_period_median_minutes": float("nan"),
            "holding_period_p95_minutes": float("nan"),
            "holding_period_mean_minutes_wins": float("nan"),
            "holding_period_mean_minutes_losses": float("nan"),
        }
    dur = np.asarray(dur_min[:n], dtype=float)
    Rn  = np.asarray(R[:n], dtype=float)
    wins = dur[Rn >  eps]
    losses = dur[Rn < -eps]
    return {
        "holding_period_mean_minutes": float(np.mean(dur)) if dur.size else float("nan"),
        "holding_period_median_minutes": float(np.median(dur)) if dur.size else float("nan"),
        "holding_period_p95_minutes": float(np.quantile(dur, 0.95, method='linear')) if dur.size else float("nan"),
        "holding_period_mean_minutes_wins": float(np.mean(wins)) if wins.size else float("nan"),
        "holding_period_mean_minutes_losses": float(np.mean(losses)) if losses.size else float("nan"),
    }


def _get_trades_bootstrap_seed() -> int:
    raw = _SCHEMA_CACHE.get("raw_schema", {})
    rt = (raw.get("runtime", {}) or {}).get("trades_bootstrap", {}) or {}
    sd = rt.get("seed", None)
    try:
        return int(44 if sd is None else sd)
    except Exception:
        return 44

def trades_full_period_summary(trades_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute all full-period trades metrics per JSON spec (one-row DataFrame)."""
    cols = [
        "trade_count","win_rate","profit_factor","average_winning_trade_r","average_losing_trade_r","payoff_ratio",
        "expectancy_mean_r","expectancy_median_r","r_std_dev","r_min","r_max",
        "worst_5_trade_run_r","worst_10_trade_run_r","worst_20_trade_run_r",
        "edr_100_trades_p50_r","edr_100_trades_p95_r",
        "prob_maxdd_100trades_le_5r","prob_maxdd_100trades_le_7r","prob_maxdd_100trades_le_10r",
        "losing_streak_max_p50_100trades","losing_streak_max_p95_100trades",
        "prob_losing_streak_ge_7_100trades","prob_losing_streak_ge_10_100trades",
        "holding_period_mean_minutes","holding_period_median_minutes","holding_period_p95_minutes",
        "holding_period_mean_minutes_wins","holding_period_mean_minutes_losses"
    ]
    if trades_df is None or len(trades_df) == 0:
        return pd.DataFrame([{c: (0 if c=="trade_count" else float("nan")) for c in cols}])
    eps = float(globals().get("_EPS_DEN", 1e-12))
    R, open_ts, close_ts = _tr_build_R_and_time(trades_df, float(cfg.risk_per_trade_pct), eps)
    base = _tr_basic_stats(R, eps)
    worst5  = _tr_worst_k_run(R, 5)
    worst10 = _tr_worst_k_run(R, 10)
    worst20 = _tr_worst_k_run(R, 20)
    boot = _tr_bootstrap_metrics(R, n_paths=5000, horizon=100, p=0.2, rng_seed=_get_trades_bootstrap_seed())
    hold = _tr_holding_period_minutes(trades_df, R, eps)
    row = dict(base)
    row.update({
        "worst_5_trade_run_r": worst5,
        "worst_10_trade_run_r": worst10,
        "worst_20_trade_run_r": worst20,
    })
    row.update(boot)
    row.update(hold)
    return pd.DataFrame([row])


def _format_trades_full_for_csv(df: pd.DataFrame, rounding_policy: dict) -> pd.DataFrame:
    # Ensure correct column order per schema if available
    try:
        cols = [c["name"] for c in _SCHEMA_CACHE["raw_schema"]["csv_schemas"]["compounding_eoy_soy_base_100k"]["trades_full_period_summary.csv"]["columns"]]
    except Exception:
        cols = list(df.columns)
    out = df.copy()
    out = out.reindex(columns=cols)
    # Apply schema-driven rounding policy (category-based)
    try:
        out = _apply_rounding_by_policy_for_full(out, rounding_policy)
    except Exception:
        pass
    return out



# ---------- Isolated rounding helpers (pattern-based, per-file) ----------
def _round_half_up_str(x, n):


    getcontext().prec = 28
    if x is None:
        return "NaN"
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "inf" if np.isinf(x) else "NaN"
    try:
        d = Decimal(str(x))
        q = Decimal("1") if n == 0 else Decimal("1").scaleb(-n)
        dq = d.quantize(q, rounding=ROUND_HALF_UP)
        s = str(dq)
        # Only strip minus if the rounded value is exactly zero
        if dq == Decimal("0"):
            if s.startswith("-"):
                s = s.replace("-", "")
        return s
    except Exception:
        return str(x)


def _apply_rounding_by_patterns_isolated(df: pd.DataFrame, patterns_decimals: list[tuple[list[str], int]], int_cols: set[str] = None) -> pd.DataFrame:

    out = df.copy()
    int_cols = int_cols or set()
    for col in out.columns:
        name = str(col)
        # Integers leave as-is
        if col in int_cols:
            continue
        # Find first matching pattern group (priority order as given)
        nd = None
        for pats, dec in patterns_decimals:
            if any(fnmatch.fnmatch(name, pat) for pat in pats):
                nd = dec
                break
        if nd is not None:
            out[col] = [ _round_half_up_str(v, nd) if pd.notna(v) else "NaN" for v in out[col].tolist() ]
    return out


def _apply_hardcoded_rounding_trades_full(df: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit rounding for trades_full_period_summary.csv using JSON categories.
    Produces strings with HALF_UP, 'inf'/'NaN', and -0 -> 0 normalization.
    """
    getcontext().prec = 28

    ratio_cols = ["profit_factor","payoff_ratio"]
    percent_cols = ["win_rate","prob_maxdd_100trades_le_5r","prob_maxdd_100trades_le_7r","prob_maxdd_100trades_le_10r",
                    "prob_losing_streak_ge_7_100trades","prob_losing_streak_ge_10_100trades"]
    rmult_cols = ["average_winning_trade_r","average_losing_trade_r","expectancy_mean_r","expectancy_median_r",
                  "r_std_dev","r_min","r_max","worst_5_trade_run_r","worst_10_trade_run_r","worst_20_trade_run_r",
                  "edr_100_trades_p50_r","edr_100_trades_p95_r"]
    duration_cols = ["holding_period_mean_minutes","holding_period_median_minutes","holding_period_p95_minutes",
                     "holding_period_mean_minutes_wins","holding_period_mean_minutes_losses"]
    int_cols = ["trade_count","losing_streak_max_p50_100trades","losing_streak_max_p95_100trades"]

    def qfmt(x, n):
        if x is None:
            return "NaN"
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                return "inf" if np.isinf(x) else "NaN"
        try:
            d = Decimal(str(x))
            q = Decimal("1") if n == 0 else Decimal("1").scaleb(-n)
            s = str(d.quantize(q, rounding=ROUND_HALF_UP))
            if s in ("-0", "-0.0", "-0.00", "-0.000", "-0.0000", "-0.00", "-0.00000"):
                s = s.replace("-", "")
            return s
        except Exception:
            return str(x)

    out = df.copy()
    for c in out.columns:
        if c in int_cols:
            n = 0
        elif c in ratio_cols:
            n = 2
        elif c in percent_cols:
            n = 4
        elif c in rmult_cols:
            n = 2
        elif c in duration_cols:
            n = 0
        else:
            # fallback: try to infer by name
            name = c.lower()
            if any(k in name for k in ["ratio","factor"]):
                n = 2
            elif any(k in name for k in ["prob_", "share", "rate"]):
                n = 4
            elif "minutes" in name:
                n = 0
            elif any(k in name for k in ["_r", "edr"]):
                n = 2
            else:
                n = None
        if n is not None:
            out[c] = [qfmt(v, n) if pd.notna(v) else "NaN" for v in out[c].tolist()]
    return out


def write_trades_full_period_summary_csv(df: pd.DataFrame, csv_path: Path):
    df_out = _format_trades_full_for_csv(df, _SCHEMA_CACHE.get("rounding_policy", {}))
    patterns_decimals = [
        (["*profit_factor*","*payoff*","*omega*","*tail_ratio*","*gain_to_pain*","*pain_ratio*"], 2),
        (["win_rate*","prob_*","*share*"], 4),
        (["*p_value*"], 3),
        (["*r"], 2),
        (["*minutes*"], 0),
    ]
    int_cols = set(["trade_count"]) 
    df_out = _apply_rounding_by_patterns_isolated(df_out, patterns_decimals, int_cols=int_cols)
    df_out.to_csv(csv_path, index=False)



def _round_half_up(x: float, n: int) -> float:
    """Round-half-up to n decimals and return float."""
    getcontext().prec = 28
    d = Decimal(str(x))
    q = Decimal("1") if n == 0 else Decimal("1").scaleb(-n)
    d2 = d.quantize(q, rounding=ROUND_HALF_UP)
    return float(d2)

def _fmt_value_for_csv(val, nd):
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "NaN"
    if isinstance(val, float) and (math.isinf(val)):
        return "inf" if val > 0 else "-inf"
    if nd is None:
        return val
    try:
        v = _round_half_up(float(val), nd)
        s = f"{v:.{nd}f}"
        if s.startswith("-0.") and float(s) == 0.0:
            s = "0." + s.split(".")[1]
        return s
    except Exception:
        return val

def _format_yearly_for_csv(df, rounding_policy=None):
    df_out = df.copy()
    def _digits_for(col):
        name = col.lower()
        if name in {"year","trade_count","active_months_count","negative_months_in_year","months_in_year_available","positive_months_in_year"}:
            return None
        if any(k in name for k in ["sharpe","sortino","calmar","omega","profit_factor"]):
            return 2
        if any(k in name for k in ["return","drawdown","volatility","var_","es_","share","win_rate","cagr"]):
            return 4
        return None
    for col in df_out.columns:
        nd = _digits_for(col)
        df_out[col] = df_out[col].map(lambda x: _fmt_value_for_csv(x, nd))
    return df_out

def _format_monthly_for_csv(df, rounding_policy=None):
    df_out = df.copy()
    # Only rounding policy for monthly_return -> 4 decimals; others pass-through
    if "monthly_return" in df_out.columns:
        df_out["monthly_return"] = df_out["monthly_return"].map(lambda x: _fmt_value_for_csv(x, 4))
    return df_out

def _format_full_period_for_csv(df, rounding_policy=None):
    df_out = df.copy()
    def _digits_for(col):
        name = col.lower()
        # ints pass-through
        if name in {
            "months","eom_longest_underwater_months","eom_time_to_recover_months","eom_months_since_maxdd_trough",
            "intramonth_longest_underwater_months","intramonth_time_to_recover_months","intramonth_months_since_maxdd_trough",
            "active_months_count","negative_months_count_full_period","positive_months_count_full_period",
            "zero_months_count_full_period","years_covered","best_year","worst_year",
            "max_consecutive_up_months_full_period","max_consecutive_down_months_full_period","trade_count_full_period"
        }:
            return None
        # ratios 2dp
        if any(k in name for k in [
            "sharpe","sortino_ratio_annualized_full_period","calmar_ratio_full_period","martin_ratio_full_period",
            "pain_ratio_full_period","gain_to_pain_ratio_monthly_full_period","tail_ratio_p95_p5_full_period","omega_ratio_full_period"
        ]):
            return 2
        # decimals 4dp
        if any(k in name for k in [
            "cagr_full_period","volatility_annualized_full_period","eom_max_drawdown_full_period","intramonth_max_drawdown_full_period",
            "underwater_months_share_full_period","ulcer_index_full_period","pain_index_full_period",
            "total_return_full_period","wealth_multiple_full_period","ending_nav_full_period",
            "best_month_return_full_period","worst_month_return_full_period",
            "mean_monthly_return_full_period","median_monthly_return_full_period",
            "monthly_var_95_full_period","monthly_es_95_full_period","monthly_var_99_full_period","monthly_es_99_full_period",
            "xrisk"
        ]):
            return 4
        return None
    for col in df_out.columns:
        nd = _digits_for(col)
        df_out[col] = df_out[col].map(lambda x: _fmt_value_for_csv(x, nd))
    return df_out




def write_monthly_returns_csv(monthly_df: pd.DataFrame, csv_path: Path):
    rounding_policy = _SCHEMA_CACHE.get("rounding_policy", {})
    df_out = _format_monthly_for_csv(monthly_df, rounding_policy)
    df_out = _apply_rounding_by_patterns_isolated(df_out, [(["monthly_return"], 4)], int_cols=set(["year","month"]))
    df_out.to_csv(csv_path, index=False)

def write_yearly_summary_csv(ys: pd.DataFrame, csv_path: Path):
    df_out = _format_yearly_for_csv(ys)
    patterns_decimals = [
        (["*_*ccy*","*_ccy"], 2),
        (["*sharpe*","*sortino*","*calmar*"], 2),
        (["*profit_factor*","*payoff*","*martin*","*omega*","*tail_ratio*","*gain_to_pain*","*pain_ratio*","*skew*","*kurt*"], 2),
        (["*xrisk*"], 2),
        (["*p_value*"], 3),
        (["wealth_multiple_*"] , 2),
        (["*holding_period*_minutes*"], 0),
        (["*downside_deviation*","*drawdown*","*return*","*volatility*","var_*","es_*","*share*","win_rate*","*cagr*"], 4),
    
        (["*_*ccy*","*_ccy"], 2),]
    int_cols = set(['months_in_year_available', 'year', 'positive_months_in_year', 'negative_months_in_year', 'trade_count', 'active_months_count'])
    df_out = _apply_rounding_by_patterns_isolated(df_out, patterns_decimals, int_cols=int_cols)
    df_out.to_csv(csv_path, index=False)

def write_full_period_summary_csv(fp_df: pd.DataFrame, csv_path: Path):
    rounding_policy = _SCHEMA_CACHE.get("rounding_policy", {})
    df_out = _format_full_period_for_csv(fp_df, rounding_policy)
    patterns_decimals = [
        (["*_*ccy*","*_ccy"], 2),
        (["ending_nav_full_period"], 2),
        (["*_xrisk"], 2),
        (["*p_value*"], 3),
        (["ending_nav_full_period"], 0),
        (["wealth_multiple_full_period"], 2),
        (["*sharpe*","*sortino*","*calmar*"], 2),
        (["*martin*","*gain_to_pain*","*pain_ratio*","*omega*","*tail_ratio*","*skew*","*kurt*"], 2),
        (["*cagr*","*return*","*volatility*","*drawdown*","*downside_deviation*","*share*","var_*","*_es_*","annual_return_*","prob_*"] , 4),
    
        (["*_*ccy*","*_ccy"], 2),
        (["ending_nav_full_period"], 2),]
    df_out = _apply_rounding_by_patterns_isolated(df_out, patterns_decimals, int_cols=set())
    df_out.to_csv(csv_path, index=False)



def write_confidence_intervals_csv(ci_df: pd.DataFrame, csv_path: Path) -> None:
    """
    Writer for confidence_intervals.csv (compounding).
    - Column order: try schema headers, else fallback to default order matching methodology.
    - Rounding: same hard Half-Up style as other CSVs, 4 decimals for returns/vol/DD/Var/ES/Ulcer/CAGR, 2 for ratio-like (none here), ints as-is.
    """
    if ci_df is None or len(ci_df) == 0:
        # still write header if available
        hdr = _schema_headers_for("confidence_intervals.csv")
        if hdr:
            pd.DataFrame(columns=hdr).to_csv(csv_path, index=False)
        else:
            cols = [
                "period_start_month","period_end_month","scope","year",
                "metric_key","metric_label","metric_basis","units","rounding_family",
                "method","ci_level_pct","estimate","ci_low","ci_high",
                "bootstrap_type","block_mean_length_months","block_mean_length_days","n_boot","seed",
            ]
            pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
        return

    df = ci_df.copy()

    # Enforce column order
    order = _schema_headers_for("confidence_intervals.csv")
    if not order:
        order = [
            "period_start_month","period_end_month","scope","year",
            "metric_key","metric_label","metric_basis","units","rounding_family",
            "method","ci_level_pct","estimate","ci_low","ci_high",
            "bootstrap_type","block_mean_length_months","block_mean_length_days","n_boot","seed",
        ]
    # Fill missing columns
    for c in order:
        if c not in df.columns:
            df[c] = np.nan
    # Cast integer-like columns to nullable integers to avoid .0 in CSV
    for icol in ['year','block_mean_length_months','block_mean_length_days','n_boot','seed']:
        if icol in df.columns:
            try:
                df[icol] = pd.to_numeric(df[icol], errors='coerce').astype('Int64')
            except Exception:
                pass
    df = df[order]

    # Rounding policy: reuse the hard-coded patterns approach
    patterns_decimals = [
        (["*_*ccy*","*_ccy"], 2),
        (["*sharpe*","*sortino*","*calmar*","*martin*","*omega*","*gain_to_pain*","*tail_ratio*"], 2),
        (["*cagr*","*return*","*volatility*","*drawdown*","var_*","*_es_*","*ulcer*","estimate","ci_low","ci_high"], 4),
    
        (["*_*ccy*","*_ccy"], 2),]
    int_cols = set(["year","block_mean_length_months","block_mean_length_days","n_boot","seed"])
    df_out = _apply_rounding_by_patterns_isolated(df, patterns_decimals, int_cols=int_cols)

    df_out.to_csv(csv_path, index=False)
# ---------- Yearly summary (with integrated post-process) ----------
def _yearly_summary_from_monthlies(monthly_df: pd.DataFrame, trades_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    years = sorted(monthly_df["year"].unique())
    last_year = max(years) if years else None
    rows = []
    thr_min_trades = _SCHEMA_CACHE.get("min_trades_per_year_warning", 12)
    min_active_for_metrics = _SCHEMA_CACHE.get("raw_schema", {}).get("methodology", {}) \
        .get("supplementary_active_only_metrics", {}).get("min_active_months_for_metrics", 6)

    for y in years:
        grp = monthly_df.loc[monthly_df["year"] == y]
        r = grp["monthly_return"].astype(float).to_numpy()
        act = grp["active_month"].astype(bool).to_numpy()

        months_in_year = int(len(grp))
        annual_return = float(np.prod(1.0 + r) - 1.0) if months_in_year > 0 else float("nan")

        # Drawdowns (EoM + RAW)
        eom_mdd_y, *_ = _monthly_underwater_metrics_with_trough(r)
        intramonth_mdd_y = _maxdd_from_intramonth_equity_year(equity_df, y)

        # Risk/return (monthly-based) — Sharpe uses rf_m in numerator
        vol_ann = _safe_std(r, ddof=cfg.stdev_ddof) * _ann_factor()
        sharpe_ann = _sharpe_annualized_with_rf(r, cfg.rf, ddof=cfg.stdev_ddof)
        sortino_ann, insufficient_negs_flag = _sortino_annualized(r, ddof=cfg.stdev_ddof, target_m=0.0)
        calmar = _calmar(annual_return, eom_mdd_y)

        pos_cnt = int(_classify_sign_masks(r)[0].sum())
        neg_cnt = int(_classify_sign_masks(r)[1].sum())

        # Active-only subset
        r_active = r[act]
        active_count = int(act.sum())
        active_share = float(active_count / 12.0) if months_in_year > 0 else float("nan")
        if active_count > 0:
            ann_ret_active = float(np.prod(1.0 + r_active) - 1.0)
            vol_active = _safe_std(r_active, ddof=cfg.stdev_ddof) * _ann_factor()
            sharpe_active = _sharpe_annualized_with_rf(r_active, cfg.rf, ddof=cfg.stdev_ddof)
            sortino_active, _ = _sortino_annualized(r_active, ddof=cfg.stdev_ddof, target_m=0.0)
            wm_active = float(np.prod(1.0 + r_active))
            cagr_active = float(wm_active ** (12.0 / max(active_count, 1)) - 1.0)
        else:
            ann_ret_active = float("nan"); vol_active = float("nan")
            sharpe_active = float("nan"); sortino_active = float("nan"); cagr_active = float("nan")

        # Trades (per year)
        if trades_df is not None and len(trades_df) > 0:
            sub = trades_df.loc[trades_df["close_ts_utc"].dt.year == y, ["pnl_pct"]].copy()
        else:
            sub = pd.DataFrame(columns=["pnl_pct"])
        if sub.empty:
            trade_count = 0; win_rate = float("nan"); pf = float("nan")
        else:
            R = (sub["pnl_pct"].astype(float) / 100.0) / float(cfg.risk_per_trade_pct)
            eps = _EPS_DEN
            nz = R[R.abs() > eps]
            trade_count = int(len(sub))
            if nz.empty:
                win_rate = float("nan"); pf = float("nan")
            else:
                wins = int(_classify_sign_masks(nz)[0].sum())
                total = int(len(nz))
                win_rate = float(wins / total) if total > 0 else float("nan")
                pos_sum = float(nz[nz > 0].sum()) if wins > 0 else 0.0
                neg_sum = float(-nz[nz < 0].sum()) if (total - wins) > 0 else 0.0
                if neg_sum < eps and pos_sum > eps:
                    pf = float("inf")
                elif pos_sum < eps and neg_sum > eps:
                    pf = 0.0
                elif neg_sum < eps and pos_sum < eps:
                    pf = float("nan")
                else:
                    pf = float(pos_sum / neg_sum)

        # Flags
        insufficient_months = months_in_year < 12
        insufficient_active_months = active_count < int(min_active_for_metrics)
        insufficient_trades = trade_count < thr_min_trades
        is_ytd = (y == last_year) and insufficient_months

        # ---- Integrated post-process (Omega, VaR/ES) ----
        r_finite = r[np.isfinite(r)]
        if r_finite.size == 0:
            omega = float("nan"); var95 = float("nan"); es95 = float("nan")
        else:
            pos_sum_r = float(r_finite[r_finite > 0.0].sum())
            neg_sum_r = float((-r_finite[r_finite < 0.0]).sum())
            if neg_sum_r == 0.0 and pos_sum_r > 0.0:
                omega = float("inf")
            elif pos_sum_r == 0.0 and neg_sum_r > 0.0:
                omega = 0.0
            elif pos_sum_r == 0.0 and neg_sum_r == 0.0:
                omega = float("nan")
            else:
                omega = float(pos_sum_r / neg_sum_r)
            try:
                var95 = float(np.quantile(r_finite, 0.05, method="linear"))
            except TypeError:
                var95 = float(np.percentile(r_finite, 5, interpolation="linear"))
            tail = r_finite[r_finite <= var95]
            es95 = float(tail.mean()) if tail.size > 0 else float("nan")

        row = {
            "year": int(y),
            "annual_return_calendar": annual_return,
            "eom_max_drawdown_intra_year": eom_mdd_y,
            "intramonth_max_drawdown_intra_year": intramonth_mdd_y,
            "trade_count": int(trade_count),
            "win_rate": win_rate,
            "profit_factor": pf,
            "active_months_count": active_count,
            "active_months_share": active_share,
            "annual_return_active": ann_ret_active,
            "volatility_active_annualized": vol_active,
            "sharpe_active_annualized": sharpe_active,
            "sortino_active_annualized": sortino_active,
            "cagr_active": cagr_active,
            "insufficient_months": bool(insufficient_months),
            "insufficient_active_months": bool(insufficient_active_months),
            "insufficient_negative_months": bool(insufficient_negs_flag),
            "insufficient_trades": bool(insufficient_trades),
            "volatility_annualized_year": vol_ann,
            "sharpe_ratio_annualized_year": sharpe_ann,
            "sortino_ratio_annualized_year": sortino_ann,
            "calmar_ratio_year": calmar,
            "negative_months_in_year": int(neg_cnt),
            "months_in_year_available": int(months_in_year),
            "is_ytd": bool(is_ytd),
            "positive_months_in_year": int(pos_cnt),
            "omega_ratio_year_target_0m": omega,
            "monthly_var_95_year": var95,
            "monthly_es_95_year": es95,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    YEARLY_ORDER = _schema_headers_for("yearly_summary.csv")
    if YEARLY_ORDER:
        for c in YEARLY_ORDER:
            if c not in df.columns:
                df[c] = np.nan
        df = df[YEARLY_ORDER]
    return df

# ---------- Full-period summary (full methodology) ----------
def _full_period_summary(monthly_df: pd.DataFrame, yearly_df: pd.DataFrame, trades_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    names = [
        "months","cagr_full_period","volatility_annualized_full_period","sharpe_ratio_annualized_full_period","sortino_ratio_annualized_full_period","calmar_ratio_full_period",
        "eom_max_drawdown_full_period","eom_longest_underwater_months","eom_time_to_recover_months","eom_months_since_maxdd_trough",
        "intramonth_max_drawdown_full_period","intramonth_longest_underwater_months","intramonth_time_to_recover_months","intramonth_months_since_maxdd_trough",
        "underwater_months_share_full_period","ulcer_index_full_period","martin_ratio_full_period","pain_index_full_period","pain_ratio_full_period",
        "skewness_full_period","kurtosis_excess_full_period",
        "eom_max_drawdown_full_period_xrisk","intramonth_max_drawdown_full_period_xrisk","ulcer_index_full_period_xrisk",
        "monthly_var_95_full_period","monthly_es_95_full_period","monthly_var_99_full_period","monthly_es_99_full_period",
        "monthly_var_95_full_period_xrisk","monthly_es_95_full_period_xrisk","monthly_var_99_full_period_xrisk","monthly_es_99_full_period_xrisk",
        "worst_month_return_full_period_xrisk","best_month_return_full_period_xrisk",
        "active_months_count","active_months_share","volatility_active_annualized","sharpe_active_annualized","sortino_active_annualized","cagr_active",
        "negative_months_count_full_period","total_return_full_period","wealth_multiple_full_period","ending_nav_full_period",
        "period_start_month","period_end_month","positive_months_count_full_period","best_month_return_full_period","worst_month_return_full_period",
        "mean_monthly_return_full_period","median_monthly_return_full_period","zero_months_count_full_period","years_covered",
        "best_year_return_calendar_full_period","best_year","worst_year_return_calendar_full_period","worst_year","downside_deviation_annualized_full_period",
        "newey_west_tstat_mean_monthly_return","newey_west_p_value_mean_monthly_return",
        "max_consecutive_up_months_full_period","max_consecutive_down_months_full_period",
        "omega_ratio_full_period_target_0m","gain_to_pain_ratio_monthly_full_period","tail_ratio_p95_p5_full_period",
        "trade_count_full_period",
    ]

    r = monthly_df["monthly_return"].astype(float).to_numpy()
    months = int(len(r))
    pos = int(_classify_sign_masks(r)[0].sum())
    neg = int(_classify_sign_masks(r)[1].sum())
    zero = int(_classify_sign_masks(r)[2].sum())

    # EoM underwater & dd grid
    eom_maxdd, eom_longest, eom_ttr, eom_trough_idx, eom_since_trough, dd = _monthly_underwater_metrics_with_trough(r)
    # Vol/Sharpe (sample std, ddof=cfg.stdev_ddof). Newey–West kept only for t-stat later.
    rf_m = float(cfg.rf)
    lrv = _newey_west_long_run_var(r.astype(float), int(getattr(cfg, 'nw_lag', 6)))
    sd_samp = _safe_std(r, ddof=cfg.stdev_ddof)
    inst_vol_ann = sd_samp * _ann_factor()
    mu = float(np.mean(r - rf_m)) if r.size>0 else float('nan')   # rf applied in numerator
    if not np.isfinite(sd_samp) or abs(sd_samp) < 1e-12:
        inst_sharpe = float('inf') if (np.isfinite(mu) and abs(mu) > 1e-12) else float('nan')
    else:
        inst_sharpe = (mu / sd_samp) * _ann_factor()
    sortino_ann, _ = _sortino_annualized(r, ddof=cfg.stdev_ddof, target_m=0.0)

    wm = float(np.prod(1.0 + r)) if months > 0 else float("nan")
    total_ret = wm - 1.0 if (wm==wm) else float("nan")
    inst_cagr = (wm ** (12.0 / months) - 1.0) if months > 0 and (wm==wm) else float("nan")
    inst_calmar = _calmar(inst_cagr, eom_maxdd) if (inst_cagr==inst_cagr) else float("nan")

    # Intramonth path metrics
    intramonth_maxdd = _maxdd_from_intramonth_equity_full_period(equity_df)
    intramonth_longest, intramonth_ttr, intramonth_since_trough = _intramonth_underwater_metrics_months(equity_df)

    # Underwater share / Ulcer / Pain (EoM dd grid)
    if dd.size == 0:
        uw_share = float("nan"); ulcer = float("nan"); pain = float("nan")
    else:
        neg_dd = dd[dd < 0.0]
        uw_share = float(len(neg_dd) / months) if months > 0 else float("nan")
        ulcer = float(np.sqrt(np.mean(neg_dd**2))) if neg_dd.size > 0 else float("nan")
        pain  = float(np.mean(np.abs(neg_dd))) if neg_dd.size > 0 else float("nan")
    martin = _ratio_num_den(inst_cagr, ulcer)
    pain_ratio = _ratio_num_den(inst_cagr, pain)

    # Distribution metrics on r
    if r.size > 0:
        mean_r = float(np.mean(r))
        s = float(np.std(r, ddof=1)) if r.size >= 2 else float("nan")
        if np.isfinite(s) and s > 0 and r.size >= 3:
            skew = float(np.mean(((r - mean_r)/s)**3))
        else:
            skew = float("nan")
        if np.isfinite(s) and s > 0 and r.size >= 4:
            kurt_ex = float(np.mean(((r - mean_r)/s)**4) - 3.0)
        else:
            kurt_ex = float("nan")
        # Omega & Gain-to-Pain (exclude zeros from sums)
        pos_sum = float(r[r > 0.0].sum())
        neg_sum = float(-r[r < 0.0].sum())
        if neg_sum < _EPS_DEN and pos_sum > _EPS_DEN:
            omega = float("inf"); g2p = float("inf")
        elif pos_sum < _EPS_DEN and neg_sum > _EPS_DEN:
            omega = 0.0; g2p = 0.0
        elif pos_sum < _EPS_DEN and neg_sum < _EPS_DEN:
            omega = float("nan"); g2p = float("nan")
        else:
            omega = float(pos_sum / neg_sum); g2p = float(pos_sum / neg_sum)
        # Tail ratio
        try:
            q95 = float(np.quantile(r, 0.95, method="linear"))
            q05 = float(np.quantile(r, 0.05, method="linear"))
        except TypeError:
            q95 = float(np.percentile(r, 95, interpolation="linear"))
            q05 = float(np.percentile(r, 5, interpolation="linear"))
        denom = abs(q05)
        if denom < _EPS_DEN and abs(q95) > _EPS_DEN:
            tail_ratio = float("inf")
        elif denom < _EPS_DEN and abs(q95) <= _EPS_DEN:
            tail_ratio = float("nan")
        else:
            tail_ratio = float(q95 / denom)
        # VaR/ES 95/99 (lower tail)
        var95 = q05
        es95 = float(r[r <= var95].mean()) if np.sum(r <= var95) > 0 else float("nan")
        try:
            q01 = float(np.quantile(r, 0.01, method="linear"))
        except TypeError:
            q01 = float(np.percentile(r, 1, interpolation="linear"))
        var99 = q01
        es99 = float(r[r <= var99].mean()) if np.sum(r <= var99) > 0 else float("nan")
    else:
        skew = float("nan"); kurt_ex = float("nan"); omega = float("nan"); g2p = float("nan")
        var95 = float("nan"); es95 = float("nan"); var99 = float("nan"); es99 = float("nan"); tail_ratio = float("nan")

    # ×risk conversions
    rp = float(cfg.risk_per_trade_pct) if float(cfg.risk_per_trade_pct) != 0 else float("nan")
    eom_mdd_xr = abs(eom_maxdd) / rp if np.isfinite(eom_maxdd) and np.isfinite(rp) and rp>0 else float("nan")
    intramonth_mdd_xr = abs(intramonth_maxdd) / rp if np.isfinite(intramonth_maxdd) and np.isfinite(rp) and rp>0 else float("nan")
    ulcer_xr = ulcer / rp if np.isfinite(ulcer) and np.isfinite(rp) and rp>0 else float("nan")
    var95_xr = abs(var95) / rp if np.isfinite(var95) and np.isfinite(rp) and rp>0 else float("nan")
    es95_xr  = abs(es95)  / rp if np.isfinite(es95)  and np.isfinite(rp) and rp>0 else float("nan")
    var99_xr = abs(var99) / rp if np.isfinite(var99) and np.isfinite(rp) and rp>0 else float("nan")
    es99_xr  = abs(es99)  / rp if np.isfinite(es99)  and np.isfinite(rp) and rp>0 else float("nan")
    best_xr  = abs(np.nanmax(r)) / rp if r.size>0 and np.isfinite(rp) and rp>0 else float("nan")
    worst_xr = abs(np.nanmin(r)) / rp if r.size>0 and np.isfinite(rp) and rp>0 else float("nan")

    # Active-only (monthly)
    act = monthly_df["active_month"].astype(bool).to_numpy()
    active_count = int(act.sum())
    active_share = float(active_count / months) if months > 0 else float("nan")
    r_active = r[act]
    if active_count > 0:
        sd_a = _safe_std(r_active, ddof=cfg.stdev_ddof)
        vol_a = sd_a * _ann_factor()
        sharpe_a = _sharpe_annualized_with_rf(r_active, rf_m, ddof=cfg.stdev_ddof)
        sortino_a, _ = _sortino_annualized(r_active, ddof=cfg.stdev_ddof, target_m=0.0)
        wm_a = float(np.prod(1.0 + r_active))
        cagr_a = float(wm_a ** (12.0 / max(active_count, 1)) - 1.0)
    else:
        vol_a = float("nan"); sharpe_a = float("nan"); sortino_a = float("nan"); cagr_a = float("nan")

    ending_nav = float(cfg.starting_nav * wm) if (wm==wm) else float("nan")
    years_cov = int(len(sorted(monthly_df["year"].unique())))
    if months > 0:
        start_year = int(monthly_df["year"].iloc[0]); start_month = int(monthly_df["month"].iloc[0])
        end_year = int(monthly_df["year"].iloc[-1]); end_month = int(monthly_df["month"].iloc[-1])
        period_start_month = f"{start_year:04d}-{start_month:02d}"
        period_end_month = f"{end_year:04d}-{end_month:02d}"
        best_month = float(np.nanmax(r))
        worst_month = float(np.nanmin(r))
        mean_month = float(np.mean(r))
        median_month = float(np.median(r))
    else:
        period_start_month = ""; period_end_month = ""
        best_month = float("nan"); worst_month = float("nan")
        mean_month = float("nan"); median_month = float("nan")

    if yearly_df is not None and len(yearly_df) > 0:
        yr = yearly_df.dropna(subset=["annual_return_calendar"])
        if len(yr) > 0:
            i_best = int(yr["annual_return_calendar"].idxmax()); i_worst = int(yr["annual_return_calendar"].idxmin())
            best_year_ret = float(yr.loc[i_best, "annual_return_calendar"]); best_year = int(yr.loc[i_best, "year"])
            worst_year_ret = float(yr.loc[i_worst, "annual_return_calendar"]); worst_year = int(yr.loc[i_worst, "year"])
        else:
            best_year_ret = float("nan"); best_year = int(np.nan); worst_year_ret = float("nan"); worst_year = int(np.nan)
    else:
        best_year_ret = float("nan"); best_year = int(np.nan); worst_year_ret = float("nan"); worst_year = int(np.nan)

    trade_count_full = int(len(trades_df)) if trades_df is not None else 0

    # Downside deviation (annualized, target 0)
    downside_dev_ann = float("nan")
    if r.size >= 2:
        ds = _downside_std(r, ddof=cfg.stdev_ddof, target_m=0.0)
        downside_dev_ann = ds * _ann_factor() if np.isfinite(ds) else float("nan")

    # Newey–West t-stat and p-value for mean(r)
    if months > 0 and np.isfinite(lrv) and lrv > 0:
        t_nw = float((np.sqrt(months) * np.mean(r)) / np.sqrt(lrv))
        # two-sided p-value via error function
        p_nw = float(math.erfc(abs(t_nw) / math.sqrt(2.0)))
    else:
        t_nw = float("nan"); p_nw = float("nan")

    # Serial runs
    max_up = _max_consecutive_runs(r, "up")
    max_dn = _max_consecutive_runs(r, "down")

    values = [
        months, inst_cagr, inst_vol_ann, inst_sharpe, sortino_ann, inst_calmar,
        eom_maxdd, int(eom_longest), eom_ttr, eom_since_trough,
        intramonth_maxdd, intramonth_longest, intramonth_ttr, intramonth_since_trough,
        uw_share, ulcer, martin, pain, pain_ratio,
        skew, kurt_ex,
        eom_mdd_xr, intramonth_mdd_xr, ulcer_xr,
        var95, es95, var99, es99,
        var95_xr, es95_xr, var99_xr, es99_xr,
        worst_xr, best_xr,
        active_count, active_share, vol_a, sharpe_a, sortino_a, cagr_a,
        neg, total_ret, wm, ending_nav, period_start_month, period_end_month, pos, best_month, worst_month,
        mean_month, median_month, zero, years_cov, best_year_ret, best_year, worst_year_ret, worst_year, downside_dev_ann,
        t_nw, p_nw,
        max_up, max_dn,
        omega, g2p, tail_ratio,
        trade_count_full,
    ]
    df = pd.DataFrame([dict(zip(names, values))])

    hdr = _schema_headers_for("full_period_summary.csv")
    if hdr:
        for c in hdr:
            if c not in df.columns:
                df[c] = np.nan
        df = df[hdr]
    return df



# --------------CI-----------

# ---------- CI: intramonth helpers (daily EoD) ----------
def _ci_daily_returns_and_times_from_equity(equity_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract daily simple returns and their 'to' timestamps from equity_df sorted by time_utc.
    Returns (r_daily, ts_daily). r_daily[i] corresponds to (ts[i-1] -> ts[i]) step and is aligned with ts_daily[i].
    """
    if equity_df is None or len(equity_df) == 0:
        return np.array([], dtype=float), np.array([], dtype="datetime64[ns]")
    eq = equity_df[["time_utc","capital_ccy"]].copy().sort_values("time_utc", kind="mergesort")
    vals = eq["capital_ccy"].astype(float).to_numpy()
    ts = eq["time_utc"].to_numpy()
    if vals.size < 2:
        return np.array([], dtype=float), np.array([], dtype="datetime64[ns]")
    r = vals[1:] / vals[:-1] - 1.0
    return r.astype(float), ts[1:]

def _mdd_from_nav_series(nav: np.ndarray) -> float:
    if nav is None or nav.size == 0:
        return float("nan")
    peaks = np.maximum.accumulate(nav)
    dd = nav / peaks - 1.0
    return float(np.min(dd)) if dd.size else float("nan")

def _mdd_intra_year_from_nav_and_times(nav: np.ndarray, ts: np.ndarray) -> float:
    """Max drawdown with peak reset at calendar year changes based on provided timestamps.
    Works on the resampled path (not sorted by actual time — uses sampled sequence order).
    """
    n = nav.size
    if n == 0 or ts.size != n:
        return float("nan")
    cur_peak = float(nav[0])
    cur_year = int(pd.Timestamp(ts[0]).year)
    mdd = 0.0
    for i in range(1, n):
        y = int(pd.Timestamp(ts[i]).year)
        if y != cur_year:
            cur_year = y
            cur_peak = float(nav[i])
        if nav[i] > cur_peak:
            cur_peak = float(nav[i])
        dd = float(nav[i] / cur_peak - 1.0)
        if dd < mdd:
            mdd = dd
    return float(mdd)


# ---------- CI: BCa helpers ----------

def _phi(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _phi_inv(p: float) -> float:
    """Approximate inverse CDF of standard normal (Acklam's rational approx)."""
    if not (0.0 < p < 1.0):
        if p <= 0.0: return -math.inf
        if p >= 1.0: return math.inf
    a = [ -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
           1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
           6.680131188771972e+01, -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00 ]
    plow  = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if phigh < p:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def _bca_interval(samples: np.ndarray, est: float, base_series: np.ndarray, metric_name: str, metric_fn, level_pct: int, cfg: Config) -> tuple[float,float]:
    """Compute BCa interval for one metric given bootstrap samples and jackknife on delete-1 months."""
  
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or (not math.isfinite(est)):
        return float('nan'), float('nan')
    # z0
    prop = np.mean(arr < est)
    z0 = _phi_inv(min(max(prop, 1e-12), 1 - 1e-12))
    # Jackknife delete-1
    n = base_series.size
    if n < 3:
        # fallback to percentile
        a = (100.0 - float(level_pct))/2.0
        b = 100.0 - a
        try:
            lo = float(np.quantile(arr, a/100.0, method="linear"))
            hi = float(np.quantile(arr, b/100.0, method="linear"))
        except TypeError:
            lo = float(np.percentile(arr, a, interpolation="linear"))
            hi = float(np.percentile(arr, b, interpolation="linear"))
        return lo, hi
    thetas = []
    for i in range(n):
        r_j = np.delete(base_series, i, axis=0)
        val = metric_fn(r_j, cfg).get(metric_name, float('nan'))
        thetas.append(float(val))
    thetas = np.array(thetas, dtype=float)
    thetas = thetas[np.isfinite(thetas)]
    if thetas.size < max(2, n//3):
        # insufficient jackknife stability
        a = (100.0 - float(level_pct))/2.0
        b = 100.0 - a
        try:
            lo = float(np.quantile(arr, a/100.0, method="linear"))
            hi = float(np.quantile(arr, b/100.0, method="linear"))
        except TypeError:
            lo = float(np.percentile(arr, a, interpolation="linear"))
            hi = float(np.percentile(arr, b, interpolation="linear"))
        return lo, hi
    theta_dot = float(np.mean(thetas))
    dif = theta_dot - thetas
    num = float(np.sum(dif**3))
    den = float(6.0 * (np.sum(dif**2) ** 1.5))
    acc = num/den if (np.isfinite(num) and np.isfinite(den) and abs(den) > 0.0) else 0.0
    # Adjusted alphas
    alpha = (100.0 - float(level_pct))/100.0
    a1 = alpha/2.0
    a2 = 1.0 - alpha/2.0
    z1 = _phi_inv(a1)
    z2 = _phi_inv(a2)
    def adj_alpha(z):
        denom = 1.0 - acc*(z0 + z)
        z_adj = (z0 + z)/denom if denom != 0 else (1.0 if (z0+z) > 0 else -1.0)
        alpha_star = _phi(z_adj)
        alpha_star = min(max(alpha_star, 1e-12), 1.0 - 1e-12)
        return alpha_star
    qa = adj_alpha(z1)
    qb = adj_alpha(z2)
    try:
        lo = float(np.quantile(arr, qa, method="linear"))
        hi = float(np.quantile(arr, qb, method="linear"))
    except TypeError:
        lo = float(np.percentile(arr, qa*100.0, interpolation="linear"))
        hi = float(np.percentile(arr, qb*100.0, interpolation="linear"))
    return lo, hi

# === Confidence Intervals (CI) — Registry (compounding) ===

from dataclasses import dataclass
from typing import Literal, List

MetricBasis = Literal["monthly", "eom", "intramonth"]

# === Confidence Intervals (CI) — Step 2: bases & thin wrappers ===
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Tuple
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class CIBases:
    monthly_returns: np.ndarray
    period_start_month: str
    period_end_month: str
    daily_returns: np.ndarray
    daily_times: np.ndarray

def _prepare_ci_bases(monthly_df: pd.DataFrame, equity_df: pd.DataFrame) -> CIBases:
    """Prepare base series for CI: monthly EoM returns and intramonth daily returns.
    - monthly_df is expected to contain columns ['year','month','monthly_return'].
    - equity_df is expected to contain ['time_utc','capital_ccy'].
    Returns CIBases with numpy arrays and YYYY-MM boundaries for monthly scope.
    """
    if monthly_df is None or len(monthly_df) == 0:
        mr = np.array([], dtype=float)
        start_m = ""
        end_m = ""
    else:
        cols = set(map(str, monthly_df.columns))
        # ensure sorted order by (year, month) if columns exist
        if {'year','month','monthly_return'}.issubset(cols):
            df = monthly_df.sort_values(['year','month'], kind='mergesort')
            mr = df['monthly_return'].to_numpy(dtype=float)
            try:
                start_m = _compose_year_month_yyyy_mm(int(df.iloc[0]['year']), int(df.iloc[0]['month']))
                end_m   = _compose_year_month_yyyy_mm(int(df.iloc[-1]['year']), int(df.iloc[-1]['month']))
            except Exception:
                # fallback: attempt to format from any 'period' like YYYY-MM
                if 'period' in cols:
                    start_m = str(df.iloc[0]['period'])
                    end_m   = str(df.iloc[-1]['period'])
                else:
                    start_m = ""
                    end_m = ""
        else:
            # Fallback: try generic column name for return
            ret_col = 'monthly_return' if 'monthly_return' in cols else next((c for c in monthly_df.columns if 'return' in str(c)), None)
            df = monthly_df.copy()
            mr = df[ret_col].to_numpy(dtype=float) if ret_col else np.array([], dtype=float)
            start_m = ""
            end_m = ""
    # Intramonth daily returns & their 'to' timestamps
    if equity_df is None or len(equity_df) == 0:
        rd = np.array([], dtype=float)
        ts = np.array([], dtype='datetime64[ns]')
    else:
        rd, ts = _ci_daily_returns_and_times_from_equity(equity_df)
    return CIBases(monthly_returns=mr, period_start_month=start_m, period_end_month=end_m, daily_returns=rd, daily_times=ts)

# ---- Stationary bootstrap resampling (generic, for monthly or daily bases) ----
def _resample_series_stationary(x: np.ndarray, L: int, n_boot: int, rng: np.random.Generator, times: np.ndarray | None = None) -> Iterable[Tuple[np.ndarray, np.ndarray | None]]:
    """Yield bootstrap resamples (x_boot, times_boot?) using stationary bootstrap with mean block length L.
    If times is provided, it is resampled in sync with x.
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    h = n
    for _ in range(int(n_boot)):
        idx = _stationary_bootstrap_indices(n, int(L), h, rng)
        xb = x[idx]
        tb = (times[idx] if times is not None else None)
        yield xb, tb

def _bootstrap_metric_samples(base_series: np.ndarray, metric_scalar_fn: Callable[[np.ndarray], float], L: int, n_boot: int, rng: np.random.Generator, times: np.ndarray | None = None) -> np.ndarray:
    """Return an array of bootstrap estimates for a scalar metric on base_series.
    metric_scalar_fn must accept only the series array (times-dependent metrics should be pre-sliced per-scope).
    """
    out = np.empty(int(n_boot), dtype=float)
    i = 0
    for xb, _tb in _resample_series_stationary(base_series, L, n_boot, rng, times=times):
        out[i] = float(metric_scalar_fn(xb))
        i += 1
    return out

# ---- BCa interval adapter (wraps existing _bca_interval with a local metric name) ----
def _ci_interval_scalar(samples: np.ndarray, estimate: float, base_series: np.ndarray, metric_key: str, level_pct: int, cfg) -> tuple[float, float]:
    """Compute BCa CI for a scalar metric using the module's _bca_interval.
    Adapts to the existing signature by injecting a mapping metric_fn and a global metric_name.
    """
    # mapping metric_fn expected by _bca_interval during jackknife:
    def _map_metric_fn(arr: np.ndarray, _cfg) -> dict[str, float]:
        # IMPORTANT: bind concrete scalar estimators in Step 3 before calling this function.
        raise NotImplementedError("Bind concrete scalar estimators in Step 3 before calling _ci_interval_scalar.")
    g = globals()
    prev = g.get('metric_name', None)
    g['metric_name'] = metric_key
    try:
        lo, hi = _bca_interval(np.asarray(samples, dtype=float), float(estimate), np.asarray(base_series, dtype=float), _map_metric_fn, int(level_pct), cfg)
    finally:
        if prev is None:
            g.pop('metric_name', None)
        else:
            g['metric_name'] = prev
    return float(lo), float(hi)


# === Confidence Intervals (CI) — Step 3: scalar estimators, BCa core, dispatcher, compute ===
import math
import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Any, Iterable, Tuple

# -- Scalar estimators (monthly/EoM/intramonth) --
def _est_cagr_from_monthlies(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return float("nan")
    prod = float(np.prod(1.0 + r))
    if not math.isfinite(prod) or prod <= 0.0:
        return float("nan")
    return float(prod ** (12.0 / n) - 1.0)

def _est_vol_ann_from_monthlies(r: np.ndarray, ddof: int = 1) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sd = _safe_std(r, ddof=ddof)
    return float(sd * math.sqrt(12.0))

def _est_eom_mdd_from_monthlies(r: np.ndarray) -> float:
    mdd, *_ = _monthly_underwater_metrics_with_trough(np.asarray(r, dtype=float))
    return float(mdd)

def _est_ulcer_index_from_monthlies(r: np.ndarray) -> float:
    *_, dd = _monthly_underwater_metrics_with_trough(np.asarray(r, dtype=float))
    if dd.size == 0:
        return float("nan")
    neg = dd[dd < 0.0]
    if neg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(neg * neg)))

def _est_var_from_monthlies(r: np.ndarray, q: float) -> float:
    # q = 0.95 -> 5% lower quantile; q = 0.99 -> 1% lower quantile
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    try:
        return float(np.quantile(r, 1.0 - q, method="linear"))
    except TypeError:
        return float(np.percentile(r, (1.0 - q) * 100.0, interpolation="linear"))

def _est_es_from_monthlies(r: np.ndarray, q: float) -> float:
    var = _est_var_from_monthlies(r, q)
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0 or not math.isfinite(var):
        return float("nan")
    tail = r[r <= var + 1e-12]
    if tail.size == 0:
        return float("nan")
    return float(np.mean(tail))

def _nav_from_returns(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return np.array([], dtype=float)
    return np.cumprod(1.0 + r, dtype=float)

def _est_intramonth_mdd_full(r_daily: np.ndarray) -> float:
    nav = _nav_from_returns(r_daily)
    return _mdd_from_nav_series(nav)

def _est_intramonth_mdd_intra_year(r_daily: np.ndarray, ts: np.ndarray) -> float:
    nav = _nav_from_returns(r_daily)
    return _mdd_intra_year_from_nav_and_times(nav, ts)

# -- BCa core for scalar metric (supports optional times via metric_fn signature) --
def _bca_interval_scalar(samples: np.ndarray, estimate: float, base_series: np.ndarray, metric_scalar_fn: Callable[..., float], level_pct: int, times: np.ndarray | None = None) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or (not math.isfinite(estimate)):
        return float('nan'), float('nan')
    # bias-correction z0
    prop = float(np.mean(arr < estimate))
    prop = min(max(prop, 1e-12), 1.0 - 1e-12)
    # Normal CDF/PPF helpers (reuse existing)
    z0 = _phi_inv(prop)
    # Jackknife delete-1 on base_series
    n = int(np.asarray(base_series).size)
    if n < 3:
        # Percentile fallback
        alpha = (100.0 - float(level_pct)) / 2.0
        a = alpha / 100.0
        b = 1.0 - a
        try:
            lo = float(np.quantile(arr, a, method="linear"))
            hi = float(np.quantile(arr, b, method="linear"))
        except TypeError:
            lo = float(np.percentile(arr, a*100.0, interpolation="linear"))
            hi = float(np.percentile(arr, b*100.0, interpolation="linear"))
        return lo, hi
    thetas = []
    # Build jackknife estimates
    for i in range(n):
        if times is None:
            r_j = np.delete(base_series, i, axis=0)
            try:
                val = metric_scalar_fn(r_j)
            except TypeError:
                val = metric_scalar_fn(r_j, None)
        else:
            r_j = np.delete(base_series, i, axis=0)
            ts_j = np.delete(times, i, axis=0)
            val = metric_scalar_fn(r_j, ts_j)
        thetas.append(float(val))
    thetas = np.asarray(thetas, dtype=float)
    thetas = thetas[np.isfinite(thetas)]
    if thetas.size < max(2, n // 3):
        # Percentile fallback on unstable jackknife
        alpha = (100.0 - float(level_pct)) / 2.0
        a = alpha / 100.0
        b = 1.0 - a
        try:
            lo = float(np.quantile(arr, a, method="linear"))
            hi = float(np.quantile(arr, b, method="linear"))
        except TypeError:
            lo = float(np.percentile(arr, a*100.0, interpolation="linear"))
            hi = float(np.percentile(arr, b*100.0, interpolation="linear"))
        return lo, hi
    theta_dot = float(np.mean(thetas))
    dif = theta_dot - thetas
    num = float(np.sum(dif**3))
    den = float(6.0 * (np.sum(dif**2) ** 1.5))
    acc = num/den if (math.isfinite(num) and math.isfinite(den) and abs(den) > 0.0) else 0.0
    # Adjusted alphas via BCa
    alpha = (100.0 - float(level_pct)) / 100.0
    a1 = alpha / 2.0
    a2 = 1.0 - alpha / 2.0
    z1 = _phi_inv(a1)
    z2 = _phi_inv(a2)
    def _adj_alpha(z):
        denom = 1.0 - acc * (z0 + z)
        z_adj = (z0 + z) / denom if denom != 0 else (1.0 if (z0 + z) > 0 else -1.0)
        return _phi(z_adj)
    a1s = _adj_alpha(z1)
    a2s = _adj_alpha(z2)
    try:
        lo = float(np.quantile(arr, a1s, method="linear"))
        hi = float(np.quantile(arr, a2s, method="linear"))
    except TypeError:
        lo = float(np.percentile(arr, a1s*100.0, interpolation="linear"))
        hi = float(np.percentile(arr, a2s*100.0, interpolation="linear"))
    return lo, hi

# -- Dispatcher from metric_key to scalar function --
def _ci_scalar_fn_for(metric_key: str, cfg) -> Callable[..., float]:
    if metric_key == "cagr_full_period":
        return lambda r: _est_cagr_from_monthlies(r)
    if metric_key == "volatility_annualized_full_period":
        return lambda r: _est_vol_ann_from_monthlies(r, ddof=1)
    if metric_key == "eom_max_drawdown_full_period":
        return lambda r: _est_eom_mdd_from_monthlies(r)
    if metric_key == "ulcer_index_full_period":
        return lambda r: _est_ulcer_index_from_monthlies(r)
    if metric_key == "monthly_var_95_full_period":
        return lambda r: _est_var_from_monthlies(r, 0.95)
    if metric_key == "monthly_es_95_full_period":
        return lambda r: _est_es_from_monthlies(r, 0.95)
    if metric_key == "monthly_var_99_full_period":
        return lambda r: _est_var_from_monthlies(r, 0.99)
    if metric_key == "monthly_es_99_full_period":
        return lambda r: _est_es_from_monthlies(r, 0.99)
    if metric_key == "eom_max_drawdown_intra_year":
        return lambda r: _est_eom_mdd_from_monthlies(r)
    if metric_key == "intramonth_max_drawdown_full_period":
            return lambda r, ts=None: _est_intramonth_mdd_full(r)
    if metric_key == "intramonth_max_drawdown_intra_year":
        return lambda r, ts=None: _est_intramonth_mdd_intra_year(r, ts if ts is not None else np.array([], dtype="datetime64[ns]"))
    return lambda r, *args, **kwargs: float("nan")

# -- Override bootstrap sampler to support time-aware functions --
def _bootstrap_metric_samples(base_series: np.ndarray, metric_scalar_fn: Callable[..., float], L: int, n_boot: int, rng: np.random.Generator, times: np.ndarray | None = None) -> np.ndarray:
    out = np.empty(int(n_boot), dtype=float)
    i = 0
    for xb, tb in _resample_series_stationary(base_series, L, n_boot, rng, times=times):
        try:
            if times is None:
                out[i] = float(metric_scalar_fn(xb))
            else:
                out[i] = float(metric_scalar_fn(xb, tb))
        except TypeError:
            out[i] = float(metric_scalar_fn(xb))
        i += 1
    return out

# -- Compute CI rows for the whole registry (returns a DataFrame; writer is separate) --
def compute_confidence_intervals(monthly_df: pd.DataFrame, equity_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    bases = _prepare_ci_bases(monthly_df, equity_df)
    mr = bases.monthly_returns
    rd = bases.daily_returns
    ts = bases.daily_times
    start_m = bases.period_start_month
    end_m = bases.period_end_month
    # RNG
    rng = np.random.default_rng(int(getattr(cfg, "ci_seed", 43)))
    Lm = 6  # per JSON for monthly/EoM
    Ld = 5  # per JSON for intramonth
    n_boot = int(getattr(cfg, "ci_n_boot", 5000))
    level = int(getattr(cfg, "ci_level_pct", 90))
    rows: List[Dict[str, Any]] = []
    # Build year list from monthly_df for calendar-year scopes
    years = []
    if monthly_df is not None and not monthly_df.empty and {"year","month","monthly_return"}.issubset(set(monthly_df.columns)):
        years = sorted(set(int(y) for y in monthly_df["year"].astype(int)))
    # Iterate registry
    for spec in _ci_registry_compounding():
        key = spec.metric_key
        basis = spec.metric_basis
        scope = spec.scope
        label = spec.metric_label
        units = spec.units
        rounding_family = spec.rounding_family
        method = "bootstrap_bca"
        bootstrap_type = "stationary_bootstrap"
        # Select base series per scope
        if scope == "full_period":
            if basis in ("monthly", "eom"):
                base = mr
                if base.size == 0:
                    continue
                fn = _ci_scalar_fn_for(key, cfg)
                est = float(fn(base))
                samples = _bootstrap_metric_samples(base, fn, Lm, n_boot, rng, times=None)
                lo, hi = _bca_interval_scalar(samples, est, base, fn, level, times=None)
                # normalize sign for drawdowns (policy: negative decimals)
                if key in {
                    "eom_max_drawdown_full_period",
                    "eom_max_drawdown_intra_year",
                    "intramonth_max_drawdown_full_period",
                    "intramonth_max_drawdown_intra_year",
                }:
                    if est > 0:
                        est = -est
                        lo, hi = -hi, -lo
                    elif lo > hi:
                        lo, hi = hi, lo
                row = {
                    "period_start_month": start_m,
                    "period_end_month": end_m,
                    "scope": "full_period",
                    "year": np.nan,
                    "metric_key": key,
                    "metric_label": label,
                    "metric_basis": basis,
                    "units": units,
                    "rounding_family": rounding_family,
                    "method": method,
                    "ci_level_pct": level,
                    "estimate": est,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_type": bootstrap_type,
                    "block_mean_length_months": Lm if basis in ("monthly","eom") else np.nan,
                    "block_mean_length_days": np.nan,
                    "n_boot": n_boot,
                    "seed": int(getattr(cfg, "ci_seed", 43)),
                }
                rows.append(row)
            elif basis == "intramonth":
                base = rd
                if base.size == 0:
                    continue
                fn = _ci_scalar_fn_for(key, cfg)
                est = float(fn(base, ts))
                samples = _bootstrap_metric_samples(base, fn, Ld, n_boot, rng, times=ts)
                lo, hi = _bca_interval_scalar(samples, est, base, fn, level, times=ts)
                # normalize sign for drawdowns (policy: negative decimals)
                if key in {
                    "eom_max_drawdown_full_period",
                    "eom_max_drawdown_intra_year",
                    "intramonth_max_drawdown_full_period",
                    "intramonth_max_drawdown_intra_year",
                }:
                    if est > 0:
                        est = -est
                        lo, hi = -hi, -lo
                    elif lo > hi:
                        lo, hi = hi, lo
                row = {
                    "period_start_month": start_m,
                    "period_end_month": end_m,
                    "scope": "full_period",
                    "year": np.nan,
                    "metric_key": key,
                    "metric_label": label,
                    "metric_basis": basis,
                    "units": units,
                    "rounding_family": rounding_family,
                    "method": method,
                    "ci_level_pct": level,
                    "estimate": est,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_type": bootstrap_type,
                    "block_mean_length_months": np.nan,
                    "block_mean_length_days": Ld,
                    "n_boot": n_boot,
                    "seed": int(getattr(cfg, "ci_seed", 43)),
                }
                rows.append(row)
        else:  # scope == "calendar_year"
            for y in years:
                if basis in ("monthly","eom"):
                    sub = monthly_df.loc[monthly_df["year"].astype(int) == int(y)].sort_values(["year","month"], kind="mergesort")
                    base = sub["monthly_return"].to_numpy(dtype=float) if not sub.empty else np.array([], dtype=float)
                    if base.size == 0:
                        continue
                    fn = _ci_scalar_fn_for(key, cfg)
                    est = float(fn(base))
                    # fresh RNG per year for determinism
                    rng_y = np.random.default_rng(int(getattr(cfg, "ci_seed", 43)))
                    samples = _bootstrap_metric_samples(base, fn, Lm, n_boot, rng_y, times=None)
                    lo, hi = _bca_interval_scalar(samples, est, base, fn, level, times=None)
                    # normalize sign for drawdowns (policy: negative decimals)
                    if key in {
                        "eom_max_drawdown_full_period",
                        "eom_max_drawdown_intra_year",
                        "intramonth_max_drawdown_full_period",
                        "intramonth_max_drawdown_intra_year",
                    }:
                        if est > 0:
                            est = -est
                            lo, hi = -hi, -lo
                        elif lo > hi:
                            lo, hi = hi, lo
                    row = {
                        "period_start_month": start_m,
                        "period_end_month": end_m,
                        "scope": "calendar_year",
                        "year": int(y),
                        "metric_key": key,
                        "metric_label": label,
                        "metric_basis": basis,
                        "units": units,
                        "rounding_family": rounding_family,
                        "method": method,
                        "ci_level_pct": level,
                        "estimate": est,
                        "ci_low": lo,
                        "ci_high": hi,
                        "bootstrap_type": bootstrap_type,
                        "block_mean_length_months": Lm,
                        "block_mean_length_days": np.nan,
                        "n_boot": n_boot,
                        "seed": int(getattr(cfg, "ci_seed", 43)),
                    }
                    rows.append(row)
                elif basis == "intramonth":
                    if rd.size == 0:
                        continue
                    years_arr = pd.to_datetime(ts).to_series(index=np.arange(len(ts))).dt.year.to_numpy(); mask = (years_arr == int(y))
                    base = rd[mask]
                    tsub = ts[mask]
                    if base.size == 0:
                        continue
                    fn = _ci_scalar_fn_for(key, cfg)
                    est = float(fn(base, tsub))
                    rng_y = np.random.default_rng(int(getattr(cfg, "ci_seed", 43)))
                    samples = _bootstrap_metric_samples(base, fn, Ld, n_boot, rng_y, times=tsub)
                    lo, hi = _bca_interval_scalar(samples, est, base, fn, level, times=tsub)
                    # normalize sign for drawdowns (policy: negative decimals)
                    if key in {
                        "eom_max_drawdown_full_period",
                        "eom_max_drawdown_intra_year",
                        "intramonth_max_drawdown_full_period",
                        "intramonth_max_drawdown_intra_year",
                    }:
                        if est > 0:
                            est = -est
                            lo, hi = -hi, -lo
                        elif lo > hi:
                            lo, hi = hi, lo
                    row = {
                        "period_start_month": start_m,
                        "period_end_month": end_m,
                        "scope": "calendar_year",
                        "year": int(y),
                        "metric_key": key,
                        "metric_label": label,
                        "metric_basis": basis,
                        "units": units,
                        "rounding_family": rounding_family,
                        "method": method,
                        "ci_level_pct": level,
                        "estimate": est,
                        "ci_low": lo,
                        "ci_high": hi,
                        "bootstrap_type": bootstrap_type,
                        "block_mean_length_months": np.nan,
                        "block_mean_length_days": Ld,
                        "n_boot": n_boot,
                        "seed": int(getattr(cfg, "ci_seed", 43)),
                    }
                    rows.append(row)
    df = pd.DataFrame(rows)
    # Final sign normalization for drawdowns (negative decimals)
    try:
        dd_keys = {
            "eom_max_drawdown_full_period",
            "eom_max_drawdown_intra_year",
            "intramonth_max_drawdown_full_period",
            "intramonth_max_drawdown_intra_year",
        }
        if not df.empty and "metric_key" in df.columns:
            msk = df["metric_key"].isin(dd_keys)
            if msk.any():
                for col in ["estimate","ci_low","ci_high"]:
                    df.loc[msk, col] = -pd.to_numeric(df.loc[msk, col], errors="coerce").abs()
                # ensure low <= high
                low = df.loc[msk, "ci_low"].to_numpy()
                high = df.loc[msk, "ci_high"].to_numpy()
                new_low = np.minimum(low, high)
                new_high = np.maximum(low, high)
                df.loc[msk, "ci_low"] = new_low
                df.loc[msk, "ci_high"] = new_high
    except Exception:
        pass
    return df

MetricScope = Literal["full_period", "calendar_year"]

@dataclass(frozen=True)
class MetricSpec:
    metric_key: str
    metric_label: str
    metric_basis: MetricBasis
    scope: MetricScope
    units: str              # e.g., "decimal"
    rounding_family: str    # e.g., "percent_like"
    estimator_id: str       # dispatch key for estimator function (to be implemented in Step 2)

def _ci_registry_compounding() -> List[MetricSpec]:
    """Registry of CI metrics for the compounding track.
    This registry is used by the CI engine to determine which metrics to compute,
    which data basis to use, and how to label/round them in the output CSV.
    """
    return [
# Monthly/EoM basis — full period
MetricSpec("cagr_full_period", "CAGR (Full Period)", "monthly", "full_period", "decimal", "percent_like", "cagr_full_period"),
MetricSpec("volatility_annualized_full_period", "Volatility Annualized (Full Period)", "monthly", "full_period", "decimal", "percent_like", "volatility_annualized_full_period"),
MetricSpec("eom_max_drawdown_full_period", "EoM Max Drawdown (Full Period)", "eom", "full_period", "decimal", "percent_like", "eom_max_drawdown_full_period"),
# ****** moved intramonth MDD (full) right after EoM MDD (full) ******
MetricSpec("intramonth_max_drawdown_full_period", "Intramonth Max Drawdown (Full Period)", "intramonth", "full_period", "decimal", "percent_like", "intramonth_max_drawdown_full_period"),
MetricSpec("ulcer_index_full_period", "Ulcer Index (Full Period)", "eom", "full_period", "decimal", "percent_like", "ulcer_index_full_period"),
MetricSpec("monthly_var_95_full_period", "Monthly VaR 95% (Full Period)", "monthly", "full_period", "decimal", "percent_like", "monthly_var_95_full_period"),
MetricSpec("monthly_es_95_full_period", "Monthly ES 95% (Full Period)", "monthly", "full_period", "decimal", "percent_like", "monthly_es_95_full_period"),
MetricSpec("monthly_var_99_full_period", "Monthly VaR 99% (Full Period)", "monthly", "full_period", "decimal", "percent_like", "monthly_var_99_full_period"),
MetricSpec("monthly_es_99_full_period", "Monthly ES 99% (Full Period)", "monthly", "full_period", "decimal", "percent_like", "monthly_es_99_full_period"),

# Monthly/EoM basis — per calendar year
MetricSpec("eom_max_drawdown_intra_year", "EoM Max Drawdown (Calendar Year)", "eom", "calendar_year", "decimal", "percent_like", "eom_max_drawdown_intra_year"),

# Intramonth basis — per calendar year
MetricSpec("intramonth_max_drawdown_intra_year", "Intramonth Max Drawdown (Calendar Year)", "intramonth", "calendar_year", "decimal", "percent_like", "intramonth_max_drawdown_intra_year"),
]


def _apply_hardcoded_rounding_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    # monthly_return -> percent_like (4), others pass-through
    patterns_decimals = [
        (["monthly_return"], 4),
    ]
    # year, month, active_month (bool) — no rounding
    return _apply_rounding_by_patterns_isolated(df, patterns_decimals, int_cols=set(["year","month"]))



def _apply_hardcoded_rounding_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Use JSON-like categories; priority order matters
    patterns_decimals = [
        (["*sharpe*","*sortino*","*calmar*"], 2),             # sharpe_sortino_like
        (["*profit_factor*","*payoff*","*martin*","*skew*","*kurt*","*omega*","*tail_ratio*","*gain_to_pain*","*pain_ratio*"], 2),  # ratio_like
        (["*xrisk*"], 2),                                      # xrisk_like
        (["*holding_period*_minutes*"], 0),                    # duration_like_minutes (in case appears)
        (["*downside_deviation*","*drawdown*","*return*","*volatility*","var_*","es_*","*share*","win_rate*","*cagr*"], 4), # percent_like superset
        (["wealth_multiple_*"], 2),                            # wealth_like
        (["*p_value*"], 3),                                    # pvalue_like
    ]
    # integer-like columns
    int_cols = set(["year","trade_count","active_months_count","months_in_year_available","positive_months_in_year","negative_months_in_year"])
    return _apply_rounding_by_patterns_isolated(df, patterns_decimals, int_cols=int_cols)


# ---------- Main ----------
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Advanced compounding metrics (clean, full methodology)")
    ap.add_argument("--input", type=str, default=None, help="Input folder (default: ./input next to this script)")
    ap.add_argument("--output", type=str, default=None, help="Output folder (default: ./output next to this script)")
    ap.add_argument("--risk-pct", type=float, default=1.0, help="Risk per trade in percent (e.g., 1.0 -> 1%)")
    ap.add_argument("--rf-m", type=float, default=0.0, help="Monthly risk-free rate (decimal, default 0.0)")
    ap.add_argument("--nw-lag", type=int, default=6, help="Newey–West lag q (used in full-period stats)")
    ap.add_argument("--start-nav", type=float, default=100000.0, help="Starting NAV for compounding (default 100000)")
    
    
    ap.add_argument("--ci-n-boot", type=int, default=None,
                        help="Override CI bootstrap sample count for faster test runs (temporary).")
    ap.add_argument("--rebase-gap-tol-abs", type=float, default=10.0,
                        help="Absolute tolerance (in account currency) to treat Dec→Jan jump as cash flow (default 10).")
    ap.add_argument("--no-rebase", dest="rebase_enable", action="store_false",
                        help="Disable year-boundary rebasing (Dec→Jan).")
    ap.add_argument("--no-rebase-write-log", dest="rebase_write_log", action="store_false",
                        help="Do not write rebasing_applied.csv audit log.")
    ap.set_defaults(rebase_enable=True, rebase_write_log=True)
    ap.add_argument("--mc-method", type=str, default=None, choices=["stationary_bootstrap","moving_block_bootstrap"], help="MC method (default from JSON)")
    ap.add_argument("--mc-block-L", type=str, default=None, help="Block mean length L in months (int or comma-list), default from JSON")
    ap.add_argument("--mc-horizons", type=str, default=None, help="Horizons in months (e.g. 12,36,full_period), default from JSON")
    ap.add_argument("--mc-paths", type=int, default=None, help="Monte Carlo number of paths (default from JSON)")
    ap.add_argument("--mc-seed", type=int, default=None, help="Monte Carlo RNG seed (optional, default from JSON)")
    ap.add_argument("--mc-dd-mode", type=str, default=None, choices=["normalized","absolute","both"], help="EoM drawdown threshold mode (default from JSON)")
    ap.add_argument("--mc-dd-thresholds", type=str, default=None, help="Comma list of EoM DD thresholds for probabilities (default from JSON)")
    
    # --- Policy-aware MC (Variant A) ---
    ap.add_argument('--mc-policy', choices=['none', 'cap_nav'], default='none',
                    help='Policy-aware Monte Carlo: none (default) or cap_nav (apply a yearly/monthly NAV cap per simulated path).')
    ap.add_argument('--mc-cap-nav-ccy', type=float, default=None,
                    help='NAV cap in account currency; required when --mc-policy=cap_nav.')
    ap.add_argument('--mc-policy-apply', choices=['yearly', 'monthly'], default='yearly',
                    help='When to apply the NAV cap: yearly (at year start, before applying that month) or monthly (after each month).')
    ap.add_argument('--mc-policy-start-month', type=int, default=1,
                    help='Month number (1..12) that anchors the year start for policy application; default=1 (January).')
    ap.add_argument('--mc-cap-tolerance-ccy', type=float, default=0.0,
                    help='Tolerance around the cap to ignore pennies (e.g., 50 for $50).')
    ap.add_argument("--tail-ratio-mode", type=str, choices=["all","active"], default="all",
                    help="Tail Ratio base: 'all' (default) or 'active' (active months only).")
    args = ap.parse_args(argv)
    # [V3] Enforce rebase ON (legacy '--no-rebase' is deprecated and ignored).
    if hasattr(args, 'no_rebase') and getattr(args, 'no_rebase'):
        _warn("[V3] '--no-rebase' is deprecated and ignored. Rebase is always ON.")
        try:
            setattr(args, 'no_rebase', False)
        except Exception:
            pass

    # [V3] Validate policy-aware MC args (no behavior change when mc_policy='none').
    if getattr(args, 'mc_policy', 'none') == 'cap_nav':
        if getattr(args, 'mc_cap_nav_ccy', None) is None or float(args.mc_cap_nav_ccy) <= 0.0:
            raise SystemExit("MC policy 'cap_nav' requires a positive --mc-cap-nav-ccy value.")
        msm = int(getattr(args, 'mc_policy_start_month', 1))
        if not (1 <= msm <= 12):
            raise SystemExit("--mc-policy-start-month must be in 1..12.")

    here = Path(__file__).resolve().parent
    in_dir = Path(args.input) if args.input else (here / "input")
    out_dir = Path(args.output) if args.output else (here / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    # [V3] Enforce rebase ON in config and map MC policy args
    try:
        if hasattr(cfg, 'rebase_enable'):
            cfg.rebase_enable = True
        if hasattr(args, 'mc_policy'): cfg.mc_policy = args.mc_policy
        if hasattr(args, 'mc_cap_nav_ccy'): cfg.mc_cap_nav_ccy = args.mc_cap_nav_ccy
        if hasattr(args, 'mc_policy_apply'): cfg.mc_policy_apply = args.mc_policy_apply
        if hasattr(args, 'mc_policy_start_month'): cfg.mc_policy_start_month = int(args.mc_policy_start_month)
        if hasattr(args, 'mc_cap_tolerance_ccy'): cfg.mc_cap_tolerance_ccy = float(args.mc_cap_tolerance_ccy)
    except Exception as _e:
        _warn(f"[V3] Failed to finalize MC policy config: {_e}")
    rp = float(args.risk_pct)
    if rp >= 1.0:
        rp /= 100.0
    cfg.risk_per_trade_pct = rp
    cfg.rf = float(args.rf_m)
    cfg.nw_lag = int(args.nw_lag)
    cfg.starting_nav = float(args.start_nav)
    try:
        cfg.tail_ratio_mode = str(getattr(args, 'tail_ratio_mode', 'all'))
    except Exception:
        cfg.tail_ratio_mode = 'all'

    # Rebasing options (default enabled; CLI may override)
    cfg.rebase_enable = bool(getattr(args, "rebase_enable", True))
    cfg.rebase_gap_tol_abs_ccy = float(getattr(args, "rebase_gap_tol_abs", 10.0))
    cfg.rebase_write_log = bool(getattr(args, "rebase_write_log", True))

    # Temporary fast-CI override (no change to defaults unless provided)
    _ci_nb = getattr(args, "ci_n_boot", None)
    if _ci_nb is not None:
        try:
            cfg.ci_n_boot = int(_ci_nb)
        except Exception:
            pass



    years = discover_years_from_input(in_dir)
    eq_parts, tr_parts = [], []
    for y in years:
        trades_path, equity_path = find_input_files_for_year(in_dir, y)
        if equity_path:
            eq_parts.append(load_equity_minimal(equity_path))
        if trades_path:
            tr_parts.append(load_trades_minimal(trades_path))

    if eq_parts:
        equity_df = pd.concat(eq_parts, ignore_index=True)

    else:
        # V9: Fallback build equity_df from balance_*.csv (month-end last) with timestamp dedup
        from pathlib import Path as _Path
        import pandas as _pd
        _parts = []
        for _p in sorted(_Path(in_dir).glob("balance_*.csv")):
            try:
                _df = _pd.read_csv(_p, sep=";")
                _df["time_utc"] = _pd.to_datetime(_df["time_utc"], utc=False, errors="coerce")
                _df = _df.dropna(subset=["time_utc"]).sort_values("time_utc")
                _df = _df.drop_duplicates(subset=["time_utc"], keep="last")
                _parts.append(_df[["time_utc","capital_ccy"]])
            except Exception:
                continue
        if _parts:
            _raw = _pd.concat(_parts, ignore_index=True).sort_values("time_utc")
            _raw = _raw.drop_duplicates(subset=["time_utc"], keep="last").set_index("time_utc")["capital_ccy"]
            _me = _raw.resample("M").last()
            equity_df = _me.to_frame(name="capital_ccy").reset_index()
        else:
            equity_df = pd.DataFrame({"time_utc": pd.to_datetime([]), "capital_ccy": []})
    # --- Year-boundary rebasing (Dec→Jan) ---
    equity_df_raw = equity_df.copy()  # keep original for flow-aware reporting if needed
    if cfg.rebase_enable:
        try:
            rebases = _detect_year_boundary_rebases_absolute(equity_df, tol_abs_ccy=cfg.rebase_gap_tol_abs_ccy)
        except Exception:
            rebases = []
        if rebases:
            equity_df = _apply_rebases_to_equity(equity_df, rebases)
            if cfg.rebase_write_log:
                outdir = Path(args.output) if hasattr(args, "output") else Path(".")
                outdir.mkdir(parents=True, exist_ok=True)
                write_rebasing_applied_csv(rebases, outdir / "rebasing_applied.csv")

    else:
        equity_df = pd.DataFrame({"time_utc": pd.to_datetime([]), "capital_ccy": []})

    if tr_parts:
        trades_df = pd.concat(tr_parts, ignore_index=True)
    else:
        trades_df = pd.DataFrame({"open_ts_utc": [], "close_ts_utc": [], "pnl_abs": [], "pnl_pct": []})

    # Monthly
    eom = _eom_nav_from_equity_no_sort(equity_df)
    mr = _compute_monthly_returns(eom, cfg.starting_nav)
    grid = mr.index
    active = _active_month_flags(trades_df, grid)

    monthly_df = pd.DataFrame({
        "year": grid.year.astype(int),
        "month": grid.month.astype(int),
        "monthly_return": mr.values.astype(float),
        "active_month": active.values.astype(bool),
    })
    write_monthly_returns_csv(monthly_df, out_dir / "monthly_returns.csv")

    # Yearly (integrated post-process inside)
    yearly_df = _yearly_summary_from_monthlies(monthly_df, trades_df, equity_df, cfg)
    
    
    # Append raw NAV bounds per calendar year (from RAW equity EoM series)
    try:
        _y_bounds = _yearly_raw_nav_bounds(equity_df_raw)
        if "year" in yearly_df.columns:
            _start = yearly_df["year"].astype(int).map({y: v.get("start") for y,v in _y_bounds.items()})
            _end   = yearly_df["year"].astype(int).map({y: v.get("end") for y,v in _y_bounds.items()})
            yearly_df["year_start_nav_ccy_raw"] = _start.astype(float).round(2)
            yearly_df["year_end_nav_ccy_raw"]   = _end.astype(float).round(2)
    except Exception:
        pass
# Append opening_cashflow_ccy (cash flow applied at the opening of the year, from prior Dec→Jan boundary)
    try:
        _oc_map = _map_opening_cashflow_by_year(locals().get("rebases", []))
        if "year" in yearly_df.columns:
            yearly_df["opening_cashflow_ccy"] = yearly_df["year"].astype(int).map(_oc_map).fillna(0.0).astype(float).round(2)
    except Exception:
        pass
    write_yearly_summary_csv(yearly_df, out_dir / "yearly_summary.csv")

    # Full period
    full_df = _full_period_summary(monthly_df, yearly_df, trades_df, equity_df, cfg)
    
    # Append flow-aware money fields (computed on RAW equity + rebases) to the end of full_df
    try:
        _flow_fields = _compute_flow_aware_full_period_fields(equity_df_raw, locals().get("rebases", []), cfg.starting_nav)
        for _k, _v in _flow_fields.items():
            full_df[_k] = _v
    except Exception:
        pass
    write_full_period_summary_csv(full_df, out_dir / "full_period_summary.csv")
    # [V8] Optional post-processing: Tail Ratio based on active-only months
    try:
        if getattr(cfg, 'tail_ratio_mode', 'all') == 'active':
            mr_path = out_dir / "monthly_returns.csv"
            fp_path = out_dir / "full_period_summary.csv"
            if mr_path.exists() and fp_path.exists():
                _mr = pd.read_csv(mr_path)
                _r_all = _mr["monthly_return"].astype(float).to_numpy()
                _act = None
                try:
                    _act = _mr["active_month"].astype(int).to_numpy()
                except Exception:
                    pass
                _r = _r_all[_act.astype(bool)] if _act is not None and _act.size == _r_all.size and (_act.astype(bool)).any() else _r_all
                try:
                    _q95 = float(np.quantile(_r, 0.95, method="linear")); _q05 = float(np.quantile(_r, 0.05, method="linear"))
                except TypeError:
                    _q95 = float(np.percentile(_r, 95, interpolation="linear")); _q05 = float(np.percentile(_r, 5, interpolation="linear"))
                _den = abs(_q05)
                if _den < _EPS_DEN and abs(_q95) > _EPS_DEN:
                    _tail = float("inf")
                elif _den < _EPS_DEN and abs(_q95) <= _EPS_DEN:
                    _tail = float("nan")
                else:
                    _tail = float(_q95 / _den)
                _fp = pd.read_csv(fp_path)
                if "tail_ratio_p95_p5_full_period" in _fp.columns:
                    _fp.loc[:, "tail_ratio_p95_p5_full_period"] = _tail
                    _fp.to_csv(fp_path, index=False)
    except Exception as _e:
        _warn(f"[V8] Tail Ratio (active-only) post-process skipped: {_e}")



    tr_full = trades_full_period_summary(trades_df, cfg)
    write_trades_full_period_summary_csv(tr_full, out_dir / "trades_full_period_summary.csv")

    write_dd_quantiles_full_period_csv(monthly_df, equity_df, cfg, out_dir / "dd_quantiles_full_period.csv")
    roll12_df = _rolling_12m_metrics(monthly_df, equity_df, cfg)
    write_rolling_12m_csv(roll12_df, out_dir / "rolling_12m.csv")
    roll36_df = _rolling_36m_metrics(monthly_df, equity_df, cfg)
    write_rolling_36m_csv(roll36_df, out_dir / "rolling_36m.csv")

    # Confidence Intervals (compounding) — compute and write
    ci_df = compute_confidence_intervals(monthly_df, equity_df, cfg)
    write_confidence_intervals_csv(ci_df, out_dir / "confidence_intervals.csv")

    print("Wrote monthly_returns.csv")
    print("Wrote yearly_summary.csv")
    print("Wrote full_period_summary.csv")

    # Monte Carlo (defaults from JSON if flags omitted)
    raw = _SCHEMA_CACHE.get("raw_schema", {})
    mc_cfg = _mc_defaults_from_schema()
    # override from CLI if provided
    if args.mc_method:
        mc_cfg["method"] = args.mc_method
    if args.mc_block_L:
        lst = _mc_parse_list_str(args.mc_block_L, allow_full=False)
        if lst is not None:
            mc_cfg["block_mean_length_months"] = _mc_sanitize_Ls(lst)
    if args.mc_horizons:
        lst = _mc_parse_list_str(args.mc_horizons, allow_full=True)
        if lst is not None:
            mc_cfg["horizons_months"] = lst
    if args.mc_paths is not None:
        mc_cfg["n_paths"] = int(args.mc_paths)
    if args.mc_seed is not None:
        mc_cfg["seed"] = int(args.mc_seed)
    if args.mc_dd_mode:
        mc_cfg["dd_thresholds_eom_mode"] = args.mc_dd_mode
    if args.mc_dd_thresholds:
        lst = _mc_parse_list_str(args.mc_dd_thresholds, allow_full=False)
        if lst is not None:
            mc_cfg["dd_thresholds_eom"] = [float(x) for x in lst]

    # Normalize L to list for cartesian product
    Ls = _mc_sanitize_Ls(mc_cfg["block_mean_length_months"])  # ensure list within 3..12

    # Period bounds as YYYY-MM
    period_start_month = None
    period_end_month = None
    if "year" in monthly_df.columns and "month" in monthly_df.columns and len(monthly_df) > 0:
        y0, m0 = monthly_df.iloc[0]["year"], monthly_df.iloc[0]["month"]
        y1, m1 = monthly_df.iloc[-1]["year"], monthly_df.iloc[-1]["month"]
        period_start_month = _compose_year_month_yyyy_mm(y0, m0)
        period_end_month = _compose_year_month_yyyy_mm(y1, m1)
    elif "month" in monthly_df.columns and len(monthly_df) > 0:
        period_start_month = _fmt_month_yyyy_mm(monthly_df.iloc[0]["month"])
        period_end_month = _fmt_month_yyyy_mm(monthly_df.iloc[-1]["month"])

    n_total = len(monthly_df)
    mr = monthly_df["monthly_return"].values.astype(float)
    risk_pct = cfg.risk_per_trade_pct  # already normalized to decimal (1.0 -> 0.01)

    rows = []
    for L in Ls:
        for h in mc_cfg["horizons_months"]:
            if isinstance(h, str) and h.lower().startswith("full"):
                H = n_total
            else:
                H = int(h)
            if H <= 0 or H > n_total:
                continue  # skip horizons longer than history
            res = _compute_mc_summary_for(mr, mc_cfg["method"], int(L), H, int(mc_cfg["n_paths"]), mc_cfg["seed"], risk_pct, nav0=cfg.starting_nav)
            
            # --- Policy-aware aggregate (Variant A) computed at main level ---
            policy_agg = None
            try:
                # Use CLI args if present
                if hasattr(args, "mc_policy") and str(getattr(args, "mc_policy", "none") or "none").lower() == "cap_nav":
                    cap_val = float(getattr(args, "mc_cap_nav_ccy", 0.0) or 0.0)
                    if cap_val > 0.0:
                        policy_agg = _mc_policy_aggregate_from_specs(
                            monthly_returns=mr,
                            method=mc_cfg["method"],
                            L=int(L),
                            n_paths=int(mc_cfg["n_paths"]),
                            H=int(H),
                            seed=int(mc_cfg["seed"]),
                            nav0=float(cfg.starting_nav),
                            mc_policy="cap_nav",
                            mc_cap_nav_ccy=cap_val,
                            mc_policy_apply=str(getattr(args, "mc_policy_apply", "yearly") or "yearly"),
                            mc_policy_start_month=int(getattr(args, "mc_policy_start_month", 1) or 1),
                            mc_cap_tolerance_ccy=float(getattr(args, "mc_cap_tolerance_ccy", 0.0) or 0.0),
                        )
            except Exception:
                policy_agg = None
            row = {
                "period_start_month": period_start_month,
                "period_end_month": period_end_month,
                "method": mc_cfg["method"],
                "block_mean_length_months": int(L),
                "n_paths": int(mc_cfg["n_paths"]),
                "horizon_months": int(H),
                "seed": int(mc_cfg["seed"]),
                "risk_per_trade_pct": float(risk_pct),
            }
            
            # Attach policy config + metrics (columns appended at the end via _MC_COLUMNS_ORDER)
            if policy_agg is not None:
                row.update({
                    "mc_policy": "cap_nav",
                    "mc_cap_nav_ccy": float(getattr(args, "mc_cap_nav_ccy", 0.0) or 0.0),
                    "mc_policy_apply": str(getattr(args, "mc_policy_apply", "yearly") or "yearly"),
                    "mc_policy_start_month": int(getattr(args, "mc_policy_start_month", 1) or 1),
                })
                row.update(policy_agg)
            row.update(res)
            rows.append(row)


    if rows:
        mc_df = pd.DataFrame(rows)
        _write_monte_carlo_summary_csv(mc_df, out_dir / "monte_carlo_summary.csv")

    return 0





# --- Policy-aware MC helper (Variant A: yearly NAV cap) ---
def _mc_policy_aggregate_from_specs(monthly_returns,
                                    method: str,
                                    L: int,
                                    n_paths: int,
                                    H: int,
                                    seed: int,
                                    nav0: float,
                                    mc_policy: str,
                                    mc_cap_nav_ccy: float,
                                    mc_policy_apply: str = "yearly",
                                    mc_policy_start_month: int = 1,
                                    mc_cap_tolerance_ccy: float = 0.0):
    """
    Compute policy-aware Monte Carlo aggregate for a given spec using the existing MC helpers.

    Parameters
    ----------
    monthly_returns : pd.Series or np.ndarray
        Source monthly returns series (already rebased/stitched).
    method : {"stationary_bootstrap", "moving_block_bootstrap", ...}
        Bootstrap method to use (must be supported by _simulate_mc_paths).
    L : int
        Block length parameter for bootstrap.
    n_paths : int
        Number of MC paths.
    H : int
        Horizon length in months.
    seed : int
        Random seed.
    nav0 : float
        Starting NAV to compound.
    mc_policy : str
        Policy name; supported: "cap_nav". Any other value returns None (policy off).
    mc_cap_nav_ccy : float
        NAV cap in currency units; must be > 0 for policy to apply.
    mc_policy_apply : {"yearly","monthly"}
        When to apply the policy. For Variant A we use "yearly".
    mc_policy_start_month : int
        Month index [1..12] that denotes the beginning of policy year (1 = January).
    mc_cap_tolerance_ccy : float
        Optional tolerance when comparing NAV to cap (to ignore tiny differences).

    Returns
    -------
    dict or None
        Aggregated policy metrics (policy_total_cash_out_ccy_p05/p50/p95, etc.) or None if policy is off.
    """
    try:
        if str(mc_policy or "none").lower() != "cap_nav":
            return None
        cap_val = float(mc_cap_nav_ccy or 0.0)
        if not (cap_val > 0.0):
            return None

        # Build paths using the same generator as intrinsic MC
        paths = _simulate_mc_paths(monthly_returns, method, L, n_paths, H, seed)
        stats_per_path = []
        for i in range(n_paths):
            stats = _mc_apply_policy_cap_nav(
                path_returns=paths[i],
                starting_nav=float(nav0),
                cap_nav=cap_val,
                apply_mode=str(mc_policy_apply or "yearly"),
                start_month=int(mc_policy_start_month or 1),
                tolerance=float(mc_cap_tolerance_ccy or 0.0),
            )
            stats_per_path.append(stats)
        return _mc_aggregate_policy(stats_per_path)
    except Exception as e:
        # Be safe: never break intrinsic MC if policy aggregation fails.
        return None

if __name__ == "__main__":
    raise SystemExit(main())


    # Append policy-aware aggregate metrics if available
    if policy_agg is not None:
        out.update({
            "mc_policy": getattr(cfg, "mc_policy", "cap_nav"),
            "mc_cap_nav_ccy": float(getattr(cfg, "mc_cap_nav_ccy", float("nan"))),
            "mc_policy_apply": str(getattr(cfg, "mc_policy_apply", "yearly")),
            "mc_policy_start_month": int(getattr(cfg, "mc_policy_start_month", 1)),
        })
        out.update(policy_agg)