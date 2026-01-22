# smoke_test.py
# where there is smoke... 
from core.generators.transaction_risk_dgp import generate_transaction_risk_dataset

df = generate_transaction_risk_dataset(seed=1955)
print(df.shape, df["is_fraud"].mean())
