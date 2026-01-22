# Dataset builder
import numpy as np
import pandas as pd

def make_housing_realistic(n=900, seed=1955, noise=20000):
    """
    Generates a realistic synthetic housing dataset with intentional messiness.
    Students practice cleaning, imputing, and modeling.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Generate realistic core features
    # ---------------------------------------------------------

    # House size in square feet (normal distribution, clipped)
    sqft = np.clip(rng.normal(1800, 600, n), 450, 4200)

    # Bedrooms roughly scale with square footage
    bedrooms = np.round(sqft / 800 + rng.normal(0, 0.6, n))
    bedrooms = np.clip(bedrooms, 1, 6).astype(int)

    # Bathrooms scale with bedrooms (with noise)
    bathrooms = bedrooms - 1 + rng.normal(0.2, 0.5, n)
    bathrooms = np.clip(bathrooms.round(), 1, 4).astype(int)

    # Home age (gamma distribution = more small values, few very old homes)
    age_years = np.clip(rng.gamma(2.0, 12.0, n), 0, 120).round()

    # Lot size (acres); lognormal ensures right-skew (few very large lots)
    lot_size = rng.lognormal(mean=-2.8, sigma=0.5, size=n)
    lot_size = np.clip((sqft / 5000) + lot_size, 0.03, 1.5).round(3)

    # Distance from city center (km), right-skew (most houses farther out)
    dist_to_center_km = np.clip(rng.lognormal(2.3, 0.5, n), 0.5, 45).round(2)

    # ---------------------------------------------------------
    # 2. Construct true price using nonlinear + linear effects
    # ---------------------------------------------------------

    base_price = 75_000
    price = (
        base_price
        + 130 * sqft
        + 12_000 * bedrooms
        + 9_000 * bathrooms
        - 400 * age_years
        + 85_000 * lot_size
        - 2_800 * dist_to_center_km
        + 0.02 * sqft / np.sqrt(dist_to_center_km)    # small nonlinearity
        + rng.normal(0, noise + 0.02 * sqft, n)       # heteroscedastic noise
    )

    df = pd.DataFrame({
        "sqft": sqft.round(0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years.astype(int),
        "lot_size": lot_size,
        "dist_to_center_km": dist_to_center_km,
        "price": price.round(0),
    })

    # ---------------------------------------------------------
    # 3. Inject messiness for cleaning practice
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed + 1)

    # Missing values
    df.loc[rng.random(n) < 0.08, "sqft"] = np.nan
    df.loc[rng.random(n) < 0.05, "lot_size"] = np.nan

    # Impossible sqft & distance values to catch
    bad_idx = df.sample(3, random_state=seed + 2).index
    df.loc[bad_idx, "sqft"] = [0, -50, 12000]  # invalid small/negative or huge values
    df.loc[df.sample(2, random_state=seed + 3).index, "dist_to_center_km"] = [0, 999]

    # Missing target values (small %)
    df.loc[rng.random(n) < 0.02, "price"] = np.nan

    return df


def make_auto_mpg_realistic(n=1200, seed=1955, noise=2.5):
    """
    Generates a realistic auto MPG dataset modeled after the classic Auto MPG dataset.
    Includes intentional missing values and invalid entries for cleaning practice.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Generate core automotive features
    # ---------------------------------------------------------

    # Engine horsepower (roughly normal, clipped)
    horsepower = np.clip(rng.normal(120, 35, n), 40, 320).round()

    # Engine displacement (correlated with horsepower)
    displacement = np.clip(rng.normal(200, 60, n), 70, 450).round(1)

    # Vehicle weight (heavier vehicles -> lower MPG)
    weight = np.clip(rng.normal(3200, 600, n), 1800, 5500).round()

    # Acceleration (0–60 time proxy)
    acceleration = np.clip(rng.normal(15, 3, n), 6, 30).round(1)

    # Model year (1970–1982)
    model_year = rng.integers(1970, 1983, n)

    # Region of vehicle origin
    origin = rng.choice(["USA", "EU", "JPN"], n, p=[0.55, 0.25, 0.20])

    # ---------------------------------------------------------
    # 2. Construct true MPG using realistic relationships
    # ---------------------------------------------------------

    mpg = (
        55
        - 0.01 * horsepower   # more HP → lower MPG
        - 0.004 * weight      # heavier → lower MPG
        + 0.7 * acceleration  # faster acceleration → slightly higher MPG (proxy)
        + 0.03 * (model_year - 1970)  # gradual improvements over time
        + (origin == "JPN") * 2.0     # origin effect
        + (origin == "EU") * 1.0
        + rng.normal(0, noise, n)     # moderate noise
    )

    df = pd.DataFrame({
        "horsepower": horsepower,
        "displacement": displacement,
        "weight": weight,
        "acceleration": acceleration,
        "model_year": model_year,
        "origin": origin,
        "mpg": mpg.round(1),
    })

    # ---------------------------------------------------------
    # 3. Inject messiness for cleaning practice
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed + 1)

    df.loc[rng.random(n) < 0.08, "horsepower"] = np.nan
    df.loc[rng.random(n) < 0.05, "weight"] = np.nan
    df.loc[rng.random(n) < 0.03, "origin"] = None

    # Impossible weights / mpg to force cleaning & diagnostics
    bad = df.sample(3, random_state=seed + 2).index
    df.loc[bad, "weight"] = [0, -100, 12000]
    df.loc[df.sample(2, random_state=seed + 3).index, "mpg"] = [1.0, 120.0]

    return df

