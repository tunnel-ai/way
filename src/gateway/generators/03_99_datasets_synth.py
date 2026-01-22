# === Module 3 Dataset Helpers ===
import numpy as np
import pandas as pd

def make_heart_disease_synth(n=600, seed=1955):
    """
    Synthetic heart disease dataset.
    Binary classification: disease (1) vs no disease (0).
    Features loosely modeled on common risk factors.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(29, 80, n)                  # years
    sex = rng.integers(0, 2, n)                    # 0=female, 1=male
    chol = np.clip(rng.normal(240, 40, n), 150, 400)   # cholesterol
    trestbps = np.clip(rng.normal(130, 15, n), 90, 200) # resting bp
    maxhr = np.clip(rng.normal(150, 25, n), 80, 210)    # max heart rate
    smoker = rng.integers(0, 2, n)                 # 0/1
    diabetic = rng.integers(0, 2, n)               # 0/1
    family_hist = rng.integers(0, 2, n)            # 0/1
    ex_angina = rng.integers(0, 2, n)              # exercise induced angina (0/1)

    # Build a logistic risk score
    z = (
        -6.0
        + 0.04 * age
        + 0.8 * sex
        + 0.015 * (chol - 200)
        + 0.02 * (trestbps - 120)
        - 0.01 * (maxhr - 140)
        + 0.9 * smoker
        + 1.0 * diabetic
        + 0.7 * family_hist
        + 0.8 * ex_angina
        + rng.normal(0, 0.7, n)   # noise
    )

    # Logistic link
    p = 1 / (1 + np.exp(-z))
    disease = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cholesterol": chol.round(0),
        "rest_bp": trestbps.round(0),
        "max_hr": maxhr.round(0),
        "smoker": smoker,
        "diabetic": diabetic,
        "family_history": family_hist,
        "exercise_angina": ex_angina,
        "disease": disease
    })

    # Messiness: missing and slightly out-of-range values
    df.loc[rng.random(n) < 0.05, "cholesterol"] = np.nan
    df.loc[rng.random(n) < 0.04, "rest_bp"] = np.nan
    df.loc[rng.random(n) < 0.03, "max_hr"] = np.nan

    return df


def make_spam_synth(n=1000, seed=1955):
    """
    Synthetic spam dataset for binary text-ish classification.
    Features represent engineered text statistics (not raw text).
    """
    rng = np.random.default_rng(seed)

    # Engineered features
    num_links = rng.poisson(1.2, n)
    num_caps = rng.poisson(3.0, n)
    num_words = rng.integers(10, 200, n)
    spammy_words = rng.poisson(0.8, n)
    from_free_domain = rng.integers(0, 2, n)
    has_reply_to = rng.integers(0, 2, n)
    exclamations = rng.poisson(0.5, n)

    # Build a probability of spam
    z = (
        -2.0
        + 0.6 * num_links
        + 0.4 * spammy_words
        + 0.7 * from_free_domain
        + 0.3 * exclamations
        + 0.1 * (num_caps - 3)
        - 0.003 * (num_words - 80)
        + rng.normal(0, 0.8, n)
    )

    p = 1 / (1 + np.exp(-z))
    spam = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "num_links": num_links,
        "num_caps": num_caps,
        "num_words": num_words,
        "spammy_words": spammy_words,
        "from_free_domain": from_free_domain,
        "has_reply_to": has_reply_to,
        "exclamations": exclamations,
        "spam": spam
    })

    # Messiness: some missing counts
    df.loc[rng.random(n) < 0.03, "num_caps"] = np.nan
    df.loc[rng.random(n) < 0.03, "spammy_words"] = np.nan

    return df


def make_wine_synth(n=800, seed=1955):
    """
    Synthetic multiclass classification dataset (wine type).
    Classes: 0 = Red, 1 = White, 2 = Rosé.
    """
    rng = np.random.default_rng(seed)

    # Common wine features
    fixed_acidity = np.clip(rng.normal(7.0, 1.2, n), 4, 15)
    volatile_acidity = np.clip(rng.normal(0.5, 0.1, n), 0.1, 1.2)
    citric_acid = np.clip(rng.normal(0.3, 0.1, n), 0.0, 1.0)
    residual_sugar = np.clip(rng.normal(6.0, 3.0, n), 0.5, 30)
    chlorides = np.clip(rng.normal(0.045, 0.02, n), 0.01, 0.2)
    free_sulfur = np.clip(rng.normal(30, 10, n), 3, 80)
    total_sulfur = np.clip(rng.normal(115, 35, n), 10, 300)
    density = np.clip(rng.normal(0.994, 0.003, n), 0.990, 1.004)
    pH = np.clip(rng.normal(3.2, 0.2, n), 2.8, 3.8)
    sulphates = np.clip(rng.normal(0.5, 0.1, n), 0.3, 1.2)
    alcohol = np.clip(rng.normal(11.0, 1.0, n), 8, 15)

    # Simple logits for 3 classes
    z_red = (
        0.8 * fixed_acidity
        + 1.2 * volatile_acidity
        + 0.4 * sulphates
        - 0.8 * alcohol
        + rng.normal(0, 0.5, n)
    )
    z_white = (
        -0.3 * volatile_acidity
        + 0.5 * residual_sugar
        + 0.3 * alcohol
        - 0.2 * sulphates
        + rng.normal(0, 0.5, n)
    )
    z_rose = (
        0.2 * residual_sugar
        + 0.2 * sulphates
        + 0.3 * alcohol
        + rng.normal(0, 0.5, n)
    )

    Z = np.vstack([z_red, z_white, z_rose]).T
    # softmax
    expZ = np.exp(Z - Z.max(axis=1, keepdims=True))
    probs = expZ / expZ.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=p) for p in probs])

    df = pd.DataFrame({
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur,
        "total_sulfur_dioxide": total_sulfur,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "wine_type": labels  # 0=red,1=white,2=rosé
    })

    # Messiness: some missing values in key columns
    df.loc[rng.random(n) < 0.03, "citric_acid"] = np.nan
    df.loc[rng.random(n) < 0.03, "residual_sugar"] = np.nan

    return df
