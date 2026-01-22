"""
transaction_risk_dgp.py

Deterministic synthetic data generator for a transaction-level "risk & loss" dataset.
Designed to support a multi-module ML course (foundations, regression, classification,
unsupervised/anomaly) with realistic properties:

- High-cardinality merchant_id sampled from a Zipf/power-law distribution
- Mixed feature types (numeric, categorical, boolean flags, text)
- Explicit MNAR missingness mechanisms (device_type, merchant_category, merchant_description)
- Two legitimate targets:
    * is_fraud (binary classification; imbalanced; nonlinear interactions + noise)
    * transaction_loss_amount (continuous regression; zero-inflated; heavy-tailed)
- Latent account "regimes" to induce discoverable structure for clustering/anomaly detection
- Optional post-event leakage fields (chargeback_flag, manual_review_score) for teaching leakage

Primary entry point:
    generate_transaction_risk_dataset(seed=1955, config=TransactionRiskConfig())

This module is intended to be imported from course notebooks, e.g.:

    from generators.transaction_risk_dgp import generate_transaction_risk_dataset
    df = generate_transaction_risk_dataset(seed=1955)

Notes:
- Generation is deterministic for a given seed + config.
- No external data downloads or authentication required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd


MERCHANT_CATEGORIES = [
    "grocery", "gas", "electronics", "travel", "luxury", "restaurant", "subscription",
    "gaming", "pharmacy", "utilities", "rideshare", "marketplace", "digital_goods",
    "apparel", "home_improvement", "financial_services", "charity", "education"
]

CHANNELS = ["card_present", "online", "mobile", "sms", "ivr"]

COUNTRIES = ["US", "CA", "MX", "GB", "FR", "DE", "IN", "BR", "JP", "AU"]  # US is domestic baseline

ACCOUNT_REGIMES = [
    "low_spend_low_freq",
    "mid_spend_mid_freq",
    "high_spend_low_freq",
    "microtxn_high_freq",
    "travel_heavy",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class TransactionRiskConfig:
    """
    Configuration for the transaction risk DGP.

    For course use, keep the schema stable and adjust *difficulty knobs* here.
    """
    # dataset size
    n_accounts: int = 6000
    target_n_transactions: int = 120_000
    n_merchants: int = 5000

    # targets
    fraud_rate_target: float = 0.04  # mean P(is_fraud=1)
    nonfraud_loss_rate: float = 0.02  # chance of loss when not fraud
    fraud_loss_event_rate: float = 0.80  # fraction of fraud that results in realized loss

    # high-cardinality merchant distribution
    zipf_s: float = 1.10  # tail heaviness (>1)

    # missingness mechanisms
    missingness_strength: float = 1.0  # scales MNAR logits

    # fraud model
    interaction_strength: float = 0.8  # foreign*new_device*online interaction multiplier
    noise_sigma: float = 1.0  # additive noise in fraud logit (higher => harder)

    # numeric caps / sanity
    max_minutes_gap: int = 7 * 24 * 60  # cap time_since_last_transaction
    max_amount: float = 5000.0          # cap transaction amounts

    # optional fields for teaching leakage
    include_leakage_fields: bool = True

    # optional: include latent probability for diagnostics
    include_latent_probability: bool = True


def make_merchants(config: TransactionRiskConfig, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create a merchant universe with Zipfian transaction volume (merchant_sampling_weight),
    low-to-moderate merchant risk variation, and imperfect metadata completeness.
    """
    m = config.n_merchants
    ranks = np.arange(1, m + 1, dtype=int)

    # Zipf-like weights over ranks (head dominates, long tail persists)
    weights = 1.0 / (ranks ** config.zipf_s)
    weights = weights / weights.sum()

    # stable primary category per merchant
    cat_idx = rng.integers(0, len(MERCHANT_CATEGORIES), size=m)
    merchant_category = np.array(MERCHANT_CATEGORIES, dtype=object)[cat_idx]

    # modest category base risk (kept imperfect intentionally)
    cat_base_risk = {
        "grocery": -0.25, "gas": -0.10, "electronics": 0.15, "travel": 0.20, "luxury": 0.30,
        "restaurant": -0.10, "subscription": 0.05, "gaming": 0.20, "pharmacy": -0.05,
        "utilities": -0.20, "rideshare": 0.05, "marketplace": 0.10, "digital_goods": 0.20,
        "apparel": 0.05, "home_improvement": 0.00, "financial_services": 0.10, "charity": -0.15,
        "education": -0.10
    }
    w_cat = np.array([cat_base_risk[c] for c in merchant_category], dtype=float)

    # merchant-specific risk is intentionally small variance to avoid pure memorization
    merchant_risk = rng.normal(0.0, 0.15, size=m) + 0.35 * w_cat

    # recognizable head merchant names + synthetic long tail
    head_names = [
        "AMAZON", "UBER", "WALMART", "APPLE", "GOOGLE", "NETFLIX", "SPOTIFY", "AIRBNB",
        "DELTA", "SHELL", "EXXON", "TARGET", "COSTCO", "BESTBUY", "PAYPAL"
    ]
    names = np.array([f"MERCH_{i:04d}" for i in range(1, m + 1)], dtype=object)
    names[:min(len(head_names), m)] = head_names[:min(len(head_names), m)]

    # metadata completeness worse in the tail (realistic nuisance)
    meta_quality = np.clip(0.85 + 0.10 * (np.log(m) - np.log(ranks)) / np.log(m), 0.55, 0.95)

    return pd.DataFrame({
        "merchant_id": np.arange(1, m + 1, dtype=int),
        "merchant_name": names,
        "merchant_category_primary": merchant_category,
        "merchant_category_risk_weight": w_cat,
        "merchant_risk": merchant_risk,
        "merchant_sampling_weight": weights,
        "merchant_meta_quality": meta_quality,
        "merchant_rank": ranks,
    })


def make_accounts(config: TransactionRiskConfig, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create accounts with latent behavioral regimes to induce cluster structure.
    """
    n = config.n_accounts

    # Regime mixture: discoverable structure for clustering / anomaly detection
    regime_probs = np.array([0.30, 0.35, 0.12, 0.15, 0.08], dtype=float)
    regime = rng.choice(ACCOUNT_REGIMES, size=n, p=regime_probs)

    account_age_days = rng.integers(30, 3650, size=n)  # ~1 month to 10 years
    base_risk = rng.normal(0, 1, size=n)

    # Foreign propensity (travel_heavy higher)
    foreign_alpha = np.where(regime == "travel_heavy", 3.0, 1.8)
    foreign_beta = np.where(regime == "travel_heavy", 5.0, 10.0)
    foreign_propensity = rng.beta(foreign_alpha, foreign_beta)

    # Spending level (log-scale mean) + variability by regime
    mu0 = np.where(regime == "low_spend_low_freq", 3.0,
           np.where(regime == "mid_spend_mid_freq", 3.5,
           np.where(regime == "high_spend_low_freq", 4.0,
           np.where(regime == "microtxn_high_freq", 2.6, 3.6))))
    mu_amt = rng.normal(mu0, 0.35)

    sigma0 = np.where(regime == "microtxn_high_freq", 0.55,
              np.where(regime == "high_spend_low_freq", 0.75, 0.65))
    sigma_amt = np.clip(rng.normal(sigma0, 0.12), 0.35, 1.05)

    # Baseline transaction rate lambda (per minute)
    lam0 = np.where(regime == "low_spend_low_freq", 1/2400,      # ~1.7/day
            np.where(regime == "mid_spend_mid_freq", 1/900,      # ~1.6/hour
            np.where(regime == "high_spend_low_freq", 1/3600,    # ~0.4/day
            np.where(regime == "microtxn_high_freq", 1/250,      # ~5.8/hour
            1/1200))))                                          # ~1.2/day
    lambda_per_min = lam0 * rng.lognormal(0.0, 0.35, size=n)

    return pd.DataFrame({
        "account_id": np.arange(1, n + 1, dtype=int),
        "account_regime": regime,
        "account_age_days": account_age_days,
        "account_base_risk": base_risk,
        "account_foreign_propensity": foreign_propensity,
        "account_mu_log_amount": mu_amt,
        "account_sigma_log_amount": sigma_amt,
        "account_lambda_per_min": lambda_per_min,
    })


def _allocate_transaction_counts(
    config: TransactionRiskConfig,
    rng: np.random.Generator,
    n_accounts: int,
) -> np.ndarray:
    """
    Allocate number of transactions per account using a gamma-poisson mixture
    to produce a heavy-tailed activity distribution.
    """
    mean_tx = config.target_n_transactions / n_accounts
    k = 1.5  # shape: smaller => heavier tail
    gamma_rates = rng.gamma(shape=k, scale=mean_tx/k, size=n_accounts)
    n_tx = rng.poisson(gamma_rates) + 1

    total = int(n_tx.sum())
    if total < int(0.8 * config.target_n_transactions) or total > int(1.2 * config.target_n_transactions):
        scale = config.target_n_transactions / max(total, 1)
        n_tx = rng.poisson(gamma_rates * scale) + 1
    return n_tx.astype(int)


def generate_transaction_risk_dataset(
    seed: int = 1955,
    config: TransactionRiskConfig = TransactionRiskConfig(),
    return_tables: bool = False,
    cache_csv: Optional[Union[str, Path]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Generate a deterministic synthetic dataset.

    Parameters
    ----------
    seed:
        Random seed for deterministic generation.
    config:
        TransactionRiskConfig controlling dataset size and difficulty knobs.
    return_tables:
        If True, also return a dict of related tables: merchants and accounts.
    cache_csv:
        Optional path to write the generated transactions table as a CSV.
        (Not required for reproducibility; primarily for inspection/debugging.)

    Returns
    -------
    df_transactions OR (df_transactions, tables)
        df_transactions includes both targets: 'is_fraud' and 'transaction_loss_amount'.
        tables = {"merchants": df_merchants, "accounts": df_accounts} if return_tables=True.
    """
    rng = np.random.default_rng(seed)

    merchants = make_merchants(config, rng)
    accounts = make_accounts(config, rng)

    n_tx_per_account = _allocate_transaction_counts(config, rng, len(accounts))
    account_ids = np.repeat(accounts["account_id"].to_numpy(), n_tx_per_account)
    tx_n = account_ids.size

    # Join account parameters by indexing (account_id starts at 1)
    acc_idx = account_ids - 1
    acc_mu = accounts["account_mu_log_amount"].to_numpy()[acc_idx]
    acc_sigma = accounts["account_sigma_log_amount"].to_numpy()[acc_idx]
    acc_lambda = accounts["account_lambda_per_min"].to_numpy()[acc_idx]
    acc_age = accounts["account_age_days"].to_numpy()[acc_idx]
    acc_risk = accounts["account_base_risk"].to_numpy()[acc_idx]
    acc_foreign = accounts["account_foreign_propensity"].to_numpy()[acc_idx]
    acc_regime = accounts["account_regime"].to_numpy()[acc_idx]

    # Time since last transaction (minutes): exponential with per-account lambda
    delta_minutes = rng.exponential(scale=1.0 / np.clip(acc_lambda, 1e-9, None), size=tx_n)
    delta_minutes = np.clip(delta_minutes, 1.0, config.max_minutes_gap)

    # Velocity proxies (noisy approximations)
    intensity_24h = np.clip(acc_lambda * 60 * 24 * (1.0 + 0.8 * np.exp(-delta_minutes/120.0)), 0.05, 80)
    tx_last_24h = rng.poisson(intensity_24h)
    intensity_7d = np.clip(acc_lambda * 60 * 24 * 7 * (1.0 + 0.3 * np.exp(-delta_minutes/720.0)), 0.2, 500)
    tx_last_7d = rng.poisson(intensity_7d)

    # Hour-of-day mixture (day vs night), regime influences night activity
    night_bias = np.where(acc_regime == "microtxn_high_freq", 0.18,
                  np.where(acc_regime == "travel_heavy", 0.14, 0.10))
    is_night = rng.random(tx_n) < night_bias
    hour = np.empty(tx_n, dtype=int)
    hour[~is_night] = rng.integers(8, 20, size=(~is_night).sum())
    hour[is_night] = rng.choice([0,1,2,3,4,5,21,22,23], size=is_night.sum())

    # Amounts: lognormal with heavy tails via Student-t noise
    eps = rng.standard_t(df=5, size=tx_n)
    amount = np.exp(acc_mu + acc_sigma * eps)
    amount = np.clip(amount, 0.5, config.max_amount)

    # Merchant sampling: draw merchant_id from Zipf weights
    merchant_id = rng.choice(
        merchants["merchant_id"].to_numpy(),
        size=tx_n,
        p=merchants["merchant_sampling_weight"].to_numpy()
    )
    merch_idx = merchant_id - 1

    merchant_name = merchants["merchant_name"].to_numpy()[merch_idx].astype(str)
    merchant_category_primary = merchants["merchant_category_primary"].to_numpy()[merch_idx].astype(object)
    merchant_risk = merchants["merchant_risk"].to_numpy()[merch_idx]
    merchant_meta_quality = merchants["merchant_meta_quality"].to_numpy()[merch_idx]
    merchant_rank = merchants["merchant_rank"].to_numpy()[merch_idx]
    cat_risk = merchants["merchant_category_risk_weight"].to_numpy()[merch_idx]

    # Foreign flag + country
    travelish = np.isin(merchant_category_primary, ["travel", "luxury", "rideshare"])
    p_foreign = np.clip(acc_foreign + 0.08 * travelish.astype(float), 0.0, 0.85)
    is_foreign = rng.random(tx_n) < p_foreign
    country = np.where(is_foreign, rng.choice(COUNTRIES[1:], size=tx_n), "US").astype(object)

    # Channel depends on category + some foreign/behavior influence
    base_channel = rng.choice(CHANNELS, size=tx_n, p=[0.45, 0.28, 0.18, 0.06, 0.03]).astype(object)
    online_boost = np.isin(merchant_category_primary, ["digital_goods", "subscription", "marketplace", "gaming"])
    card_boost = np.isin(merchant_category_primary, ["grocery", "gas", "restaurant", "pharmacy"])
    u = rng.random(tx_n)
    base_channel = np.where(online_boost & (u < 0.35), "online", base_channel)
    base_channel = np.where(card_boost & (u < 0.30), "card_present", base_channel)
    base_channel = np.where(is_foreign & (u < 0.12), "online", base_channel)
    channel = base_channel.astype(object)

    # Device type depends on channel
    device_type = np.empty(tx_n, dtype=object)
    mask = channel == "card_present"
    device_type[mask] = "pos_terminal"
    mask = channel == "online"
    device_type[mask] = rng.choice(["desktop_web", "mobile_web"], size=mask.sum(), p=[0.55, 0.45])
    mask = channel == "mobile"
    device_type[mask] = "mobile_app"
    mask = (channel == "sms") | (channel == "ivr")
    device_type[mask] = "unknown_device"

    # New device: more likely for newer accounts and non-card-present channels
    p_new_dev = 1.0 / (1.0 + np.exp(-(-1.8 + 0.9*(channel != "card_present").astype(float) - 0.0005*acc_age)))
    is_new_device = rng.random(tx_n) < p_new_dev

    # High-risk merchant flag (weak indicator)
    is_high_risk_merchant = (merchant_risk > np.quantile(merchant_risk, 0.85))

    # -------------------------
    # MNAR missingness mechanisms
    # -------------------------
    # Device type missing more often for sms/ivr, tail merchants (lower meta), and older accounts.
    logit_miss_dev = (
        -2.0
        + 1.7*(channel == "sms").astype(float)
        + 1.4*(channel == "ivr").astype(float)
        + 0.6*(channel == "mobile").astype(float)
        + 0.5*(merchant_meta_quality < 0.70).astype(float)
        + 0.35*np.log1p(acc_age)
    ) * config.missingness_strength
    p_miss_dev = _sigmoid(logit_miss_dev)
    miss_dev = rng.random(tx_n) < p_miss_dev
    device_type = device_type.astype(object)
    device_type[miss_dev] = np.nan

    # Merchant category missing more often for tail merchants and foreign transactions
    logit_miss_cat = (
        -2.6
        + 1.2*(merchant_rank > 2500).astype(float)
        + 1.0*(merchant_rank > 4500).astype(float)
        + 0.8*is_foreign.astype(float)
        + 0.6*(channel == "sms").astype(float)
    ) * config.missingness_strength
    p_miss_cat = _sigmoid(logit_miss_cat)
    miss_cat = rng.random(tx_n) < p_miss_cat
    merchant_category = merchant_category_primary.copy().astype(object)
    merchant_category[miss_cat] = np.nan

    # Text description present but sometimes missing (tail merchants / low metadata quality)
    chan_token = np.where(
        channel == "online", "ONLINE",
        np.where(channel == "card_present", "POS",
        np.where(channel == "mobile", "MOB",
        np.where(channel == "sms", "SMS", "IVR")))
    ).astype(str)
    cat_or_unknown = np.where(pd.isna(merchant_category), "UNKNOWN", merchant_category.astype(str)).astype(str)

    # robust vectorized concatenation using np.char (avoids dtype edge cases)
    base_desc = np.char.add(np.char.add(merchant_name, " "), cat_or_unknown)
    noise_tokens = rng.choice(["", "", "", " LTD", " INC", " CO", " *", " -X", " SVCS", " PAY"], size=tx_n).astype(str)
    merchant_description = np.char.add(np.char.add(np.char.add(base_desc, " "), chan_token), noise_tokens).astype(object)

    logit_miss_desc = (-2.2 + 1.1*(merchant_rank > 4200).astype(float) + 0.9*(merchant_meta_quality < 0.65).astype(float))
    p_miss_desc = _sigmoid(logit_miss_desc)
    miss_desc = rng.random(tx_n) < p_miss_desc
    merchant_description[miss_desc] = np.nan

    # -------------------------
    # Fraud probability model (logit) with interactions + noise
    # -------------------------
    ln_amt = np.log1p(amount)
    ln_vel = np.log1p(tx_last_24h)

    foreign = is_foreign.astype(float)
    newdev = is_new_device.astype(float)
    online = (channel == "online").astype(float)
    mobile = (channel == "mobile").astype(float)
    sms = (channel == "sms").astype(float)
    night = is_night.astype(float)

    interaction = foreign * newdev * online  # key nonlinear signal

    s = (
        -3.2
        + 0.55*acc_risk
        + 0.55*foreign
        + 0.45*newdev
        + 0.40*online
        + 0.20*mobile
        + 0.35*sms
        + 0.20*night
        + 0.25*ln_vel
        + 0.18*ln_amt
        + 0.45*cat_risk
        + 0.55*merchant_risk
        + config.interaction_strength * 0.9 * interaction
        + rng.normal(0.0, config.noise_sigma, size=tx_n)
    )

    # Calibrate intercept shift to match target fraud rate in expectation
    delta = 0.0
    for _ in range(10):
        p = _sigmoid(s + delta)
        f = p.mean() - config.fraud_rate_target
        fp = np.mean(p * (1 - p)) + 1e-9
        delta -= f / fp
    p_fraud = _sigmoid(s + delta)
    is_fraud = rng.random(tx_n) < p_fraud

    # -------------------------
    # Loss generation: zero-inflated, heavy-tailed
    # -------------------------
    fraud_loss_event = is_fraud & (rng.random(tx_n) < config.fraud_loss_event_rate)  # blocked fraud => zero loss
    q = rng.beta(2.0, 6.0, size=tx_n)  # mean ~0.25 of amount
    fraud_loss = q * amount * rng.lognormal(mean=-0.05, sigma=0.35, size=tx_n)

    nonfraud_loss_event = (~is_fraud) & (rng.random(tx_n) < config.nonfraud_loss_rate)
    nonfraud_loss = np.minimum(amount, rng.lognormal(mean=2.0, sigma=0.7, size=tx_n))

    transaction_loss_amount = np.zeros(tx_n, dtype=float)
    transaction_loss_amount[fraud_loss_event] = fraud_loss[fraud_loss_event]
    transaction_loss_amount[nonfraud_loss_event] = nonfraud_loss[nonfraud_loss_event]

    # Leakage-like post-event signals (optional; useful for teaching)
    if config.include_leakage_fields:
        chargeback_flag = (transaction_loss_amount > 0) & (rng.random(tx_n) < 0.85)
        manual_review_score = np.clip(p_fraud + rng.normal(0, 0.15, size=tx_n), 0, 1)
    else:
        chargeback_flag = np.zeros(tx_n, dtype=bool)
        manual_review_score = np.full(tx_n, np.nan)

    # Approximate 30d stats as noisy transforms of latent per-account parameters
    avg_transaction_amount_30d = np.clip(np.exp(acc_mu + rng.normal(0, 0.20, size=tx_n)), 0.5, config.max_amount)
    std_transaction_amount_30d = np.clip(np.exp(np.log(8.0) + rng.normal(0, 0.55, size=tx_n)), 0.1, 500.0)

    # Assemble dataframe (schema intentionally stable across modules)
    df = pd.DataFrame({
        "transaction_id": np.arange(1, tx_n + 1, dtype=int),
        "account_id": account_ids.astype(int),

        "merchant_id": merchant_id.astype(int),
        "merchant_name": merchant_name,
        "merchant_category": merchant_category,
        "payment_channel": channel,
        "country": country,

        "is_foreign_transaction": is_foreign.astype(int),
        "device_type": device_type,
        "is_new_device": is_new_device.astype(int),
        "is_high_risk_merchant": is_high_risk_merchant.astype(int),

        "hour_of_day": hour.astype(int),
        "time_since_last_transaction_minutes": np.round(delta_minutes, 2),
        "transactions_last_24h": tx_last_24h.astype(int),
        "transactions_last_7d": tx_last_7d.astype(int),

        "transaction_amount": np.round(amount, 2),
        "avg_transaction_amount_30d": np.round(avg_transaction_amount_30d, 2),
        "std_transaction_amount_30d": np.round(std_transaction_amount_30d, 2),

        "merchant_description": merchant_description,

        # targets
        "is_fraud": is_fraud.astype(int),
        "transaction_loss_amount": np.round(transaction_loss_amount, 2),
    })

    if config.include_leakage_fields:
        df["chargeback_flag"] = chargeback_flag.astype(int)
        df["manual_review_score"] = np.round(manual_review_score, 3)

    if config.include_latent_probability:
        df["fraud_probability_latent"] = np.round(p_fraud, 4)

    if cache_csv is not None:
        cache_path = Path(cache_csv)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    if return_tables:
        return df, {"merchants": merchants, "accounts": accounts}
    return df


def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick numeric summary (useful for sanity checks in instructor notebooks).
    """
    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "accounts": int(df["account_id"].nunique()),
        "merchants_used": int(df["merchant_id"].nunique()),
        "fraud_rate": float(df["is_fraud"].mean()),
        "loss_rate_any": float((df["transaction_loss_amount"] > 0).mean()),
        "avg_loss": float(df["transaction_loss_amount"].mean()),
        "avg_loss_given_loss": float(df.loc[df["transaction_loss_amount"] > 0, "transaction_loss_amount"].mean()),
        "pct_missing_device_type": float(df["device_type"].isna().mean()) if "device_type" in df.columns else float("nan"),
        "pct_missing_merchant_category": float(df["merchant_category"].isna().mean()) if "merchant_category" in df.columns else float("nan"),
        "pct_missing_description": float(df["merchant_description"].isna().mean()) if "merchant_description" in df.columns else float("nan"),
    }
    return summary
