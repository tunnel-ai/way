# === Module 4 Dataset Helpers ===
import numpy as np
import pandas as pd


def make_blobs_synth(n=500, seed=1955):
    """
    Synthetic 2D dataset for clustering.
    Designed to show:
    - Non-spherical clusters
    - Unequal densities
    - Noise points
    """
    rng = np.random.default_rng(seed)

    c1 = rng.normal(loc=[0, 0], scale=[1.0, 0.4], size=(200, 2))
    c2 = rng.normal(loc=[5, 5], scale=[0.5, 1.3], size=(150, 2))
    c3 = rng.normal(loc=[-4, 4], scale=[1.4, 0.6], size=(120, 2))

    X = np.vstack([c1, c2, c3])

    # Add uniform noise
    n_noise = int(0.06 * len(X))
    noise = rng.uniform(low=-8, high=8, size=(n_noise, 2))
    X = np.vstack([X, noise])

    df = pd.DataFrame(X, columns=["x1", "x2"])

    # Messiness: missing values
    df.loc[rng.random(len(df)) < 0.04, "x2"] = np.nan

    return df


def make_high_dim_synth(n=600, seed=1955):
    """
    Synthetic high-dimensional dataset for
    PCA / t-SNE visualization and structure discovery.
    """
    rng = np.random.default_rng(seed)

    base = rng.normal(0, 1, size=(n, 1))

    features = np.hstack([
        base + rng.normal(0, 0.2, size=(n, 1)),
        0.6 * base + rng.normal(0, 0.3, size=(n, 1)),
        rng.normal(5, 2, size=(n, 1)),
        rng.normal(-3, 1, size=(n, 1)),
        rng.exponential(1.0, size=(n, 1)),
        rng.normal(0, 0.05, size=(n, 1))  # near-constant
    ])

    df = pd.DataFrame(
        features,
        columns=[
            "spend_score",
            "engagement",
            "income_proxy",
            "debt_proxy",
            "activity_rate",
            "almost_constant"
        ]
    )

    # Inject outliers
    outliers = rng.choice(df.index, size=18, replace=False)
    df.loc[outliers, "spend_score"] *= 5

    # Messiness: missing values
    for col in df.columns:
        df.loc[rng.random(n) < 0.02, col] = np.nan

    return df


def make_fraud_synth(n=2000, anomaly_frac=0.03, seed=1955):
    """
    Synthetic anomaly detection dataset.
    Rare anomalies with subtle shifts in feature distributions.
    """
    rng = np.random.default_rng(seed)

    n_anom = int(n * anomaly_frac)
    n_norm = n - n_anom

    normal = rng.normal(0, 1, size=(n_norm, 6))
    anomalies = rng.normal(0, 3.5, size=(n_anom, 6))

    X = np.vstack([normal, anomalies])
    y = np.array([0] * n_norm + [1] * n_anom)

    df = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(6)]
    )

    # Shuffle
    df["is_anomaly"] = y
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    labels = df.pop("is_anomaly")

    # Messiness
    df.loc[rng.random(len(df)) < 0.015, "feature_3"] = np.nan

    return df, labels
