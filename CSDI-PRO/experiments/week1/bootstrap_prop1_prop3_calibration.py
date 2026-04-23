"""B2: numerical calibration of Prop 1 constant C_1 + Prop 3 rate + bootstrap CI
for Ours S0→S3 degradation.

Three analyses:
  (i) Prop 1 ambient dimension tax: empirically fit C_1 in
      NRMSE_Panda(s, σ) ≈ C_1 · √(D/n_eff) · (1 + OOD_term)
      using Panda's S0→S3 data from phase-transition main experiment.

  (ii) Prop 3 manifold posterior contraction: empirically fit the rate
       NRMSE_Ours(s, σ) ∝ n_eff^{-(2ν+1)/(2ν+1+d_KY)}
       and check whether Ours' S0→S3 -47% falls within the CI.

  (iii) Bootstrap CI for Ours' S0→S3 ratio from 5 seeds × 7 scenarios PT data.

Outputs:
  - Fitted C_1 and rate exponent
  - Bootstrap 95% CI for Ours S0→S3 ratio
  - Whether Theorem 2's quantitative closure is statistically supported
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PT_JSON = Path(__file__).resolve().parent / "results" / "pt_v2_with_panda_n5_small.json"
OUT_JSON = Path(__file__).resolve().parent / "results" / "prop1_prop3_calibration.json"


# Lorenz63 params
D_AMBIENT = 3                     # ambient dimension
D_KY_LORENZ63 = 2.06              # Kaplan-Yorke dimension
SIGMA_ATTR = 8.51                 # attractor std
NU_MATERN = 2.5                   # Matérn-5/2 kernel regularity
N_CONTEXT = 1200                  # training context length per sample

# Scenarios from PILOT_SCENARIOS (s_i, sigma_i) pairs
SCENARIO_PARAMS = {
    "S0": (0.00, 0.00),
    "S1": (0.20, 0.10),
    "S2": (0.40, 0.30),
    "S3": (0.60, 0.50),
    "S4": (0.75, 0.80),
    "S5": (0.90, 1.20),
    "S6": (0.95, 1.50),
}


def n_eff_ratio(s: float, sigma: float) -> float:
    return (1 - s) / (1 + sigma ** 2)


def load_pt_data():
    data = json.loads(PT_JSON.read_text())
    return data["records"]


def collect_method_scenario(records, method: str, scenario: str, metric: str = "rmse_norm_first100"):
    """Return list of per-seed values for (method, scenario, metric)."""
    return [r[metric] for r in records
            if r["method"] == method and r["scenario"] == scenario
            and r.get(metric) is not None and not np.isnan(r[metric])]


def bootstrap_ratio(ours_s0: list[float], ours_s3: list[float], n_boot: int = 10000,
                    rng_seed: int = 42) -> dict:
    """Bootstrap the S3/S0 VPT ratio for Ours."""
    rng = np.random.default_rng(rng_seed)
    ours_s0 = np.array(ours_s0)
    ours_s3 = np.array(ours_s3)
    # ratio of means (with resampling)
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        s0_boot = rng.choice(ours_s0, size=len(ours_s0), replace=True)
        s3_boot = rng.choice(ours_s3, size=len(ours_s3), replace=True)
        if s0_boot.mean() > 0:
            ratios[i] = s3_boot.mean() / s0_boot.mean()
        else:
            ratios[i] = np.nan
    ratios = ratios[~np.isnan(ratios)]
    return dict(
        mean=float(ratios.mean()),
        median=float(np.median(ratios)),
        ci_95_low=float(np.percentile(ratios, 2.5)),
        ci_95_high=float(np.percentile(ratios, 97.5)),
        point_estimate=float(ours_s3.mean() / ours_s0.mean()) if ours_s0.mean() > 0 else float("nan"),
    )


def fit_prop1_constant(records):
    """Fit C_1 in NRMSE_Panda ≈ C_1 · √(D/n_eff) using RMSE metric.

    Uses S0, S1 only (where Panda is in-distribution, no OOD term).
    """
    points = []
    for sc in ["S0", "S1"]:
        s, sigma = SCENARIO_PARAMS[sc]
        ratio = n_eff_ratio(s, sigma)
        n_eff = N_CONTEXT * ratio
        expected_rate = np.sqrt(D_AMBIENT / n_eff)
        rmses = collect_method_scenario(records, "panda", sc)
        if not rmses:
            continue
        for r in rmses:
            points.append((expected_rate, r, sc))

    if len(points) < 3:
        return None
    expected = np.array([p[0] for p in points])
    observed = np.array([p[1] for p in points])
    # Simple least squares: observed = C_1 * expected → C_1 = mean(observed / expected)
    C1_estimates = observed / expected
    return dict(
        C1_mean=float(C1_estimates.mean()),
        C1_std=float(C1_estimates.std()),
        C1_median=float(np.median(C1_estimates)),
        n_points=len(points),
        scenarios=list(set(p[2] for p in points)),
    )


def fit_prop3_rate(records):
    """Fit Prop 3's rate exponent.

    Prop 3: E[||K̂ - K||²] ≲ n_eff^{-(2ν+1)/(2ν+1+d_KY)}
    Taking log: log(NRMSE) ≈ const + β · log(n_eff), where β = -(2ν+1)/(2ν+1+d_KY) / 2
    (dividing by 2 because NRMSE is sqrt of the L² error).

    Theoretical β for Lorenz63 (d_KY=2.06, ν=5/2):
       rate exponent = (2·2.5+1)/(2·2.5+1+2.06) = 6/8.06 ≈ 0.744
       β = -0.744 / 2 ≈ -0.372   (NRMSE scales as n_eff^{-0.372})
    """
    # Use Ours S0-S4 (where manifold degradation is smooth and in-distribution)
    theoretical_rate = (2 * NU_MATERN + 1) / (2 * NU_MATERN + 1 + D_KY_LORENZ63)
    theoretical_beta = -theoretical_rate / 2  # ≈ -0.372

    points = []
    for sc in ["S0", "S1", "S2", "S3", "S4"]:
        s, sigma = SCENARIO_PARAMS[sc]
        n_eff = N_CONTEXT * n_eff_ratio(s, sigma)
        rmses = collect_method_scenario(records, "ours", sc)
        for r in rmses:
            points.append((np.log(n_eff), np.log(r), sc))

    if len(points) < 5:
        return None
    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])
    # Linear fit: log(rmse) = a + β log(n_eff)
    beta, a = np.polyfit(X, Y, deg=1)

    # Compute R² and residuals
    y_hat = a + beta * X
    ss_res = np.sum((Y - y_hat) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Bootstrap CI for β
    n_boot = 5000
    rng = np.random.default_rng(43)
    betas = np.empty(n_boot)
    n_p = len(points)
    for i in range(n_boot):
        idx = rng.choice(n_p, size=n_p, replace=True)
        bb, _ = np.polyfit(X[idx], Y[idx], deg=1)
        betas[i] = bb

    return dict(
        theoretical_beta=float(theoretical_beta),
        empirical_beta=float(beta),
        empirical_intercept=float(a),
        r2=float(r2),
        ci_95_beta=[float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5))],
        bootstrap_std_beta=float(betas.std()),
        matches_theoretical=bool(np.percentile(betas, 2.5) <= theoretical_beta <= np.percentile(betas, 97.5)),
        n_points=n_p,
    )


def main():
    records = load_pt_data()
    print(f"Loaded {len(records)} records from {PT_JSON.name}")

    # Use rmse_norm_first100 as the NRMSE proxy
    ours_s0 = collect_method_scenario(records, "ours", "S0")
    ours_s3 = collect_method_scenario(records, "ours", "S3")
    panda_s0 = collect_method_scenario(records, "panda", "S0")
    panda_s3 = collect_method_scenario(records, "panda", "S3")

    print(f"\nOurs S0 RMSE: {ours_s0}")
    print(f"Ours S3 RMSE: {ours_s3}")
    print(f"Panda S0 RMSE: {panda_s0}")
    print(f"Panda S3 RMSE: {panda_s3}")

    # --- (iii) Bootstrap CI for Ours S0→S3 ratio on VPT10 ---
    # Use VPT10 instead of RMSE for the Ours −47% claim (which is VPT-based)
    ours_vpt_s0 = collect_method_scenario(records, "ours", "S0", metric="vpt10")
    ours_vpt_s3 = collect_method_scenario(records, "ours", "S3", metric="vpt10")
    print(f"\nOurs VPT10 S0: {ours_vpt_s0}")
    print(f"Ours VPT10 S3: {ours_vpt_s3}")

    boot = bootstrap_ratio(ours_vpt_s0, ours_vpt_s3)
    print(f"\n=== Bootstrap: Ours S3/S0 VPT10 ratio ===")
    print(f"  point estimate = {boot['point_estimate']:.3f}  (= 1 - {(1-boot['point_estimate'])*100:.0f}%)")
    print(f"  mean = {boot['mean']:.3f}, median = {boot['median']:.3f}")
    print(f"  95% CI = [{boot['ci_95_low']:.3f}, {boot['ci_95_high']:.3f}]")
    print(f"  (translates to S0→S3 drop of [{(1-boot['ci_95_high'])*100:.0f}%, {(1-boot['ci_95_low'])*100:.0f}%])")

    # Check whether Prop 3 theoretical prediction falls within
    # Prop 3 predicts: NRMSE × (n_eff_S3/n_eff_S0)^{-0.372}
    # = NRMSE × (0.320/1.0)^{-0.372} = NRMSE × (0.320)^{-0.372}
    neff_ratio = n_eff_ratio(*SCENARIO_PARAMS["S3"]) / n_eff_ratio(*SCENARIO_PARAMS["S0"])
    prop3_predicted_vpt_ratio = neff_ratio ** 0.372  # NRMSE scales up → VPT scales down similarly
    print(f"\n  Prop 3 predicts S3/S0 VPT10 ratio ≈ {prop3_predicted_vpt_ratio:.3f} "
          f"(drop of {(1-prop3_predicted_vpt_ratio)*100:.0f}%)")
    prop3_in_ci = boot["ci_95_low"] <= prop3_predicted_vpt_ratio <= boot["ci_95_high"]
    print(f"  Prop 3 prediction inside bootstrap CI: {prop3_in_ci}")

    # --- (i) Prop 1 constant C_1 ---
    c1_fit = fit_prop1_constant(records)
    print(f"\n=== Prop 1 constant C_1 fit ===")
    if c1_fit:
        print(f"  C_1 = {c1_fit['C1_mean']:.3f} ± {c1_fit['C1_std']:.3f}  "
              f"(from {c1_fit['n_points']} points on {c1_fit['scenarios']})")

    # --- (ii) Prop 3 rate ---
    prop3_fit = fit_prop3_rate(records)
    print(f"\n=== Prop 3 rate exponent fit ===")
    if prop3_fit:
        print(f"  theoretical β (NRMSE ~ n_eff^β) = {prop3_fit['theoretical_beta']:.3f}")
        print(f"  empirical β = {prop3_fit['empirical_beta']:.3f}  (R²={prop3_fit['r2']:.3f})")
        print(f"  95% CI for β = [{prop3_fit['ci_95_beta'][0]:.3f}, {prop3_fit['ci_95_beta'][1]:.3f}]")
        print(f"  theoretical value inside empirical CI: {prop3_fit['matches_theoretical']}")

    summary = dict(
        bootstrap_s3_s0_ratio=boot,
        prop3_predicted_ratio=float(prop3_predicted_vpt_ratio),
        prop3_in_ci=bool(prop3_in_ci),
        prop1_C1=c1_fit,
        prop3_rate=prop3_fit,
        n_eff_ratios_by_scenario={sc: n_eff_ratio(*p) for sc, p in SCENARIO_PARAMS.items()},
    )
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
