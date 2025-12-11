import pandas as pd, numpy as np, random
from datetime import datetime, timedelta
from .config import RAW_DIR, PROCESSED_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def generate_synthetic_data(n_claims=2000, n_providers=80, n_users=500):
    np.random.seed(42); random.seed(42)

    providers = pd.DataFrame({
        "provider_id":[f"P{1000+i}" for i in range(n_providers)],
        "provider_name":[f"Provider_{i}" for i in range(n_providers)],
        "registration_year": np.random.randint(1995, 2024, size=n_providers),
        "license_number":[f"LIC{random.randint(10000,99999)}" for _ in range(n_providers)]
    })

    users = pd.DataFrame({
        "patient_id":[f"U{2000+i}" for i in range(n_users)],
        "age":np.random.randint(10,90,size=n_users),
        "gender":np.random.choice(['M','F','O'], size=n_users)
    })

    procedure_codes = ['PROC_A','PROC_B','PROC_C','PROC_D','PROC_XRAY','PROC_SURG']
    base_amount = {'PROC_A':1000,'PROC_B':2000,'PROC_C':500,'PROC_D':5000,'PROC_XRAY':800,'PROC_SURG':15000}

    claims = []
    start_date = datetime(2025,1,1)

    for i in range(n_claims):
        pid = random.choice(users['patient_id'])
        prov = random.choice(providers['provider_id'])
        proc = random.choice(procedure_codes)

        amt = base_amount[proc] * np.random.uniform(0.6,1.6)

        # Fraud injection
        label = 1 if random.random() < 0.025 else 0
        if label:
            amt *= np.random.uniform(3,8)

        claim_date = start_date + timedelta(days=int(np.random.randint(0,300)))

        claims.append({
            "claim_id": f"C{100000+i}",
            "patient_id": pid,
            "provider_id": prov,
            "procedure_code": proc,
            "amount": round(float(amt),2),
            "claim_date": claim_date.strftime('%Y-%m-%d'),
            "status": random.choice(['submitted','paid','denied']),
            "is_fraud_label": label
        })

    pd.DataFrame(claims).to_csv(RAW_DIR / "claims.csv", index=False)
    providers.to_csv(RAW_DIR / "providers.csv", index=False)
    users.to_csv(RAW_DIR / "users.csv", index=False)

    return {"claims": len(claims), "providers": len(providers), "users": len(users)}
