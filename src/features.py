import pandas as pd
from .config import PROCESSED_DIR

def compute_basic_features_and_stage1():
    claims = pd.read_csv(PROCESSED_DIR / "claims.csv")

    claims['amount'] = pd.to_numeric(claims['amount'], errors='coerce').fillna(0)
    claims['claim_date'] = pd.to_datetime(claims['claim_date'])

    prov_agg = claims.groupby("provider_id")['amount'].agg(['count','mean']).rename(columns={'count':'provider_total_claims','mean':'provider_mean_amount'})
    claims = claims.merge(prov_agg, on='provider_id', how='left')

    by_code = claims.groupby('procedure_code')['amount']
    q1 = by_code.quantile(0.25)
    q3 = by_code.quantile(0.75)
    iqr = q3 - q1

    claims = claims.merge(q3.rename('q3'), on='procedure_code')
    claims = claims.merge(iqr.rename('iqr'), on='procedure_code')

    claims['is_amount_outlier'] = claims['amount'] > (claims['q3'] + 3*claims['iqr'])
    claims['is_duplicate'] = claims.duplicated(subset=['patient_id','provider_id','amount','claim_date'], keep=False)

    median_claims = claims['provider_total_claims'].median()
    claims['provider_high_volume'] = claims['provider_total_claims'] > (2 * median_claims)

    claims['stage1_score'] = claims[['is_amount_outlier','is_duplicate','provider_high_volume']].astype(int).sum(axis=1)

    out = PROCESSED_DIR / "claims_stage1.parquet"
    claims.to_parquet(out, index=False)

    return {"rows": len(claims), "candidates": int((claims['stage1_score']>=1).sum())}
