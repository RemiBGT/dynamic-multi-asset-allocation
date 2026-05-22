from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


CORE_HMM_FEATURES = [
    "spy_ret_21d",
    "spy_ret_63d",
    "spy_vol_21d",
    "spy_vol_63d",
    "spy_drawdown_252d",
    "tlt_ret_21d",
    "ief_ret_21d",
    "hyg_ret_21d",
    "spy_tlt_corr_63d",
    "vix_level",
    "vix_change_21d",
    "us_10y_change_21d",
    "us_10y_2y_slope",
    "breakeven_10y_change_21d",
    "real_yield_10y_change_21d",
    "baa_10y_spread",
    "baa_10y_spread_change_21d",
    "credit_quality_spread",
]


@dataclass
class HMMFitResult:
    model: GaussianHMM
    scaler: StandardScaler
    features_scaled: np.ndarray
    states: pd.Series
    probabilities: pd.DataFrame
    log_likelihood: float
    converged: bool


def load_features(path: str | Path) -> pd.DataFrame:
    features = pd.read_csv(path, index_col="date", parse_dates=True)
    features = features.sort_index()
    return features


def load_returns(path: str | Path) -> pd.DataFrame:
    returns = pd.read_csv(path, index_col="date", parse_dates=True)
    returns = returns.sort_index()
    return returns


def select_features(
    features: pd.DataFrame,
    selected_features: Iterable[str] = CORE_HMM_FEATURES,
) -> pd.DataFrame:
    selected_features = list(selected_features)
    missing = [col for col in selected_features if col not in features.columns]

    if missing:
        raise ValueError(f"Missing features in dataframe: {missing}")

    selected = features[selected_features].copy()
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()

    return selected


def estimate_hmm_n_parameters(
    n_components: int,
    n_features: int,
    covariance_type: str = "full",
) -> int:
    """
    Approximate the number of free parameters in a Gaussian HMM.

    This is used for AIC and BIC comparison.
    """
    startprob_params = n_components - 1
    transition_params = n_components * (n_components - 1)
    mean_params = n_components * n_features

    if covariance_type == "full":
        covariance_params = n_components * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        covariance_params = n_components * n_features
    elif covariance_type == "spherical":
        covariance_params = n_components
    elif covariance_type == "tied":
        covariance_params = n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    return startprob_params + transition_params + mean_params + covariance_params


def compute_aic_bic(
    log_likelihood: float,
    n_parameters: int,
    n_observations: int,
) -> Tuple[float, float]:
    aic = 2 * n_parameters - 2 * log_likelihood
    bic = np.log(n_observations) * n_parameters - 2 * log_likelihood
    return aic, bic


def fit_gaussian_hmm(
    features: pd.DataFrame,
    n_components: int,
    covariance_type: str = "full",
    n_iter: int = 1000,
    random_seeds: Iterable[int] = (7, 21, 42, 100, 123),
) -> HMMFitResult:
    """
    Fit several Gaussian HMMs with different random seeds
    and keep the one with the highest log-likelihood.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    best_model = None
    best_score = -np.inf

    for seed in random_seeds:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=seed,
            min_covar=1e-4,
        )

        try:
            model.fit(x_scaled)
            score = model.score(x_scaled)
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError(f"Could not fit HMM with K={n_components}.")

    states_array = best_model.predict(x_scaled)
    probabilities_array = best_model.predict_proba(x_scaled)

    states = pd.Series(
        states_array,
        index=features.index,
        name="regime",
    )

    probabilities = pd.DataFrame(
        probabilities_array,
        index=features.index,
        columns=[f"prob_regime_{i}" for i in range(n_components)],
    )

    return HMMFitResult(
        model=best_model,
        scaler=scaler,
        features_scaled=x_scaled,
        states=states,
        probabilities=probabilities,
        log_likelihood=best_score,
        converged=bool(best_model.monitor_.converged),
    )


def compute_regime_feature_summary(
    features: pd.DataFrame,
    states: pd.Series,
) -> pd.DataFrame:
    aligned = features.loc[states.index].copy()
    aligned["regime"] = states

    summary = aligned.groupby("regime").agg(["mean", "std", "median"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]

    summary.insert(0, "n_obs", aligned.groupby("regime").size())
    summary["frequency"] = summary["n_obs"] / len(aligned)

    return summary


def compute_regime_return_summary(
    returns: pd.DataFrame,
    states: pd.Series,
) -> pd.DataFrame:
    common_index = returns.index.intersection(states.index)

    aligned_returns = returns.loc[common_index].copy()
    aligned_states = states.loc[common_index]

    rows = []

    for regime, group in aligned_returns.groupby(aligned_states):
        row = {
            "regime": regime,
            "n_obs": len(group),
        }

        for col in group.columns:
            annual_return = group[col].mean() * 252
            annual_volatility = group[col].std() * np.sqrt(252)

            row[f"{col}_ann_return"] = annual_return
            row[f"{col}_ann_vol"] = annual_volatility

            if annual_volatility > 0:
                row[f"{col}_ann_sharpe_0rf"] = annual_return / annual_volatility
            else:
                row[f"{col}_ann_sharpe_0rf"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("regime").sort_index()


def compute_transition_matrix_from_states(states: pd.Series) -> pd.DataFrame:
    previous_states = states.shift(1).dropna().astype(int)
    current_states = states.loc[previous_states.index].astype(int)

    matrix = pd.crosstab(
        previous_states,
        current_states,
        normalize="index",
    )

    matrix.index.name = "from_regime"
    matrix.columns.name = "to_regime"

    return matrix


def compute_regime_durations(states: pd.Series) -> pd.DataFrame:
    """
    Compute contiguous regime episode durations.
    """
    states = states.dropna().astype(int)

    change_id = (states != states.shift(1)).cumsum()

    episodes = pd.DataFrame(
        {
            "regime": states,
            "episode_id": change_id,
        },
        index=states.index,
    )

    durations = (
        episodes.groupby(["episode_id", "regime"])
        .size()
        .reset_index(name="duration_days")
    )

    summary = durations.groupby("regime")["duration_days"].agg(
        ["count", "mean", "median", "min", "max"]
    )

    summary = summary.rename(columns={"count": "n_episodes"})

    return summary


def _rank_series(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(ascending=ascending, method="average")


def suggest_regime_labels(summary: pd.DataFrame) -> Dict[int, str]:
    """
    Suggest economic labels based on regime-level feature means.

    These labels are heuristic. They are not ground truth.
    Regime IDs are arbitrary in an unsupervised HMM.
    """
    regimes = summary.index.astype(int)

    def get_col(name: str) -> pd.Series:
        col = f"{name}_mean"
        if col not in summary.columns:
            return pd.Series(0.0, index=summary.index)
        return summary[col]

    spy_ret = get_col("spy_ret_63d")
    spy_short_ret = get_col("spy_ret_21d")
    spy_vol = get_col("spy_vol_21d")
    drawdown = get_col("spy_drawdown_252d")
    tlt_ret = get_col("tlt_ret_21d")
    hyg_ret = get_col("hyg_ret_21d")
    vix = get_col("vix_level")
    vix_change = get_col("vix_change_21d")
    ten_y_change = get_col("us_10y_change_21d")
    real_y_change = get_col("real_yield_10y_change_21d")
    credit_spread = get_col("baa_10y_spread")
    credit_spread_change = get_col("baa_10y_spread_change_21d")
    equity_bond_corr = get_col("spy_tlt_corr_63d")

    risk_on_score = (
        _rank_series(spy_ret, ascending=True)
        + _rank_series(hyg_ret, ascending=True)
        + _rank_series(drawdown, ascending=True)
        + _rank_series(spy_vol, ascending=False)
        + _rank_series(vix, ascending=False)
        + _rank_series(credit_spread, ascending=False)
    )

    defensive_risk_off_score = (
        _rank_series(spy_ret, ascending=False)
        + _rank_series(spy_short_ret, ascending=False)
        + _rank_series(tlt_ret, ascending=True)
        + _rank_series(vix, ascending=True)
        + _rank_series(credit_spread, ascending=True)
        + _rank_series(credit_spread_change, ascending=True)
    )

    inflation_rates_shock_score = (
        _rank_series(tlt_ret, ascending=False)
        + _rank_series(ten_y_change, ascending=True)
        + _rank_series(real_y_change, ascending=True)
        + _rank_series(equity_bond_corr, ascending=True)
        + _rank_series(spy_ret, ascending=False)
    )

    recovery_score = (
        _rank_series(spy_short_ret, ascending=True)
        + _rank_series(hyg_ret, ascending=True)
        + _rank_series(vix_change, ascending=False)
        + _rank_series(credit_spread_change, ascending=False)
    )

    labels = {int(regime): f"mixed_or_neutral_{int(regime)}" for regime in regimes}
    remaining = set(int(regime) for regime in regimes)

    def assign_best(score: pd.Series, label: str) -> None:
        if not remaining:
            return

        candidates = score.loc[list(remaining)]
        best_regime = int(candidates.idxmax())

        labels[best_regime] = label
        remaining.remove(best_regime)

    assign_best(risk_on_score, "risk_on")
    assign_best(defensive_risk_off_score, "defensive_risk_off")
    assign_best(inflation_rates_shock_score, "inflation_rates_shock")

    if len(regimes) >= 4:
        assign_best(recovery_score, "recovery_or_normalization")

    return labels


def add_economic_labels(
    states: pd.Series,
    labels: Dict[int, str],
) -> pd.Series:
    return states.map(labels).rename("regime_label")


def build_model_selection_row(
    fit_result: HMMFitResult,
    n_components: int,
    n_features: int,
    covariance_type: str,
) -> Dict[str, float | int | bool]:
    n_observations = fit_result.features_scaled.shape[0]

    n_parameters = estimate_hmm_n_parameters(
        n_components=n_components,
        n_features=n_features,
        covariance_type=covariance_type,
    )

    aic, bic = compute_aic_bic(
        log_likelihood=fit_result.log_likelihood,
        n_parameters=n_parameters,
        n_observations=n_observations,
    )

    return {
        "n_components": n_components,
        "covariance_type": covariance_type,
        "n_observations": n_observations,
        "n_features": n_features,
        "n_parameters_approx": n_parameters,
        "log_likelihood": fit_result.log_likelihood,
        "aic": aic,
        "bic": bic,
        "converged": fit_result.converged,
    }


def get_month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Return the last available trading date of each calendar month.
    """
    month_periods = index.to_period("M")
    month_end_dates = index.to_series().groupby(month_periods).max()
    return pd.DatetimeIndex(month_end_dates.values)


def walk_forward_hmm_detection(
    features: pd.DataFrame,
    n_components: int,
    train_years: int = 5,
    covariance_type: str = "full",
    min_train_observations: int = 756,
    n_iter: int = 1000,
) -> pd.DataFrame:
    """
    Monthly walk-forward HMM regime detection.

    At each month-end date:
    - use only the last train_years of data;
    - fit scaler and HMM on that training window;
    - infer the current regime at the decision date;
    - compute one-step-ahead regime probabilities using the transition matrix.
    """
    decision_dates = get_month_end_dates(features.index)
    rows = []

    for decision_date in decision_dates:
        train_start = decision_date - pd.DateOffset(years=train_years)
        train_features = features.loc[train_start:decision_date].dropna()

        if len(train_features) < min_train_observations:
            continue

        try:
            fit_result = fit_gaussian_hmm(
                features=train_features,
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
            )
        except Exception as error:
            rows.append(
                {
                    "date": decision_date,
                    "n_components": n_components,
                    "error": str(error),
                }
            )
            continue

        current_probs = fit_result.probabilities.iloc[-1]
        current_regime = int(current_probs.values.argmax())
        current_regime_probability = float(current_probs.max())

        next_probs = current_probs.values @ fit_result.model.transmat_
        next_regime = int(np.argmax(next_probs))
        next_regime_probability = float(np.max(next_probs))

        feature_summary = compute_regime_feature_summary(
            train_features,
            fit_result.states,
        )

        labels = suggest_regime_labels(feature_summary)

        row = {
            "date": decision_date,
            "n_components": n_components,
            "train_start": train_features.index.min(),
            "train_end": train_features.index.max(),
            "train_observations": len(train_features),
            "log_likelihood": fit_result.log_likelihood,
            "converged": fit_result.converged,
            "current_regime": current_regime,
            "current_regime_label": labels.get(
                current_regime,
                f"regime_{current_regime}",
            ),
            "current_regime_probability": current_regime_probability,
            "next_regime": next_regime,
            "next_regime_label": labels.get(
                next_regime,
                f"regime_{next_regime}",
            ),
            "next_regime_probability": next_regime_probability,
            "error": "",
        }

        for i, probability in enumerate(current_probs.values):
            row[f"current_prob_regime_{i}"] = probability

        for i, probability in enumerate(next_probs):
            row[f"next_prob_regime_{i}"] = probability

        rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty and "date" in result.columns:
        result = result.sort_values("date").set_index("date")

    return result


def summarize_walk_forward_results(walk_forward_results: pd.DataFrame) -> Dict[str, float | int]:
    valid = walk_forward_results[
        walk_forward_results["error"].fillna("") == ""
    ].copy()

    if valid.empty:
        return {
            "n_decision_dates": 0,
            "mean_current_regime_probability": np.nan,
            "median_current_regime_probability": np.nan,
            "min_current_regime_probability": np.nan,
            "n_regime_switches": 0,
            "switch_rate": np.nan,
        }

    regime_changes = (
        valid["current_regime_label"] != valid["current_regime_label"].shift(1)
    )

    n_switches = int(regime_changes.iloc[1:].sum()) if len(valid) > 1 else 0

    return {
        "n_decision_dates": len(valid),
        "mean_current_regime_probability": valid[
            "current_regime_probability"
        ].mean(),
        "median_current_regime_probability": valid[
            "current_regime_probability"
        ].median(),
        "min_current_regime_probability": valid[
            "current_regime_probability"
        ].min(),
        "n_regime_switches": n_switches,
        "switch_rate": n_switches / max(len(valid) - 1, 1),
    }