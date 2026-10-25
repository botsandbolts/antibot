"""Per-source behavioral and graph-overlap features."""
import numpy as np
import pandas as pd
FEATURES=["log_events","ctr","unique_ips","ip_concentration","events_per_ip","device_diversity","campaign_diversity","slot_diversity","avg_price","hour_concentration","burst_10m_share","cross_source_ip_share"]
def maxshare(df,key,col):
    ct=df.groupby([key,col],dropna=False).size()
    return ct.groupby(level=0).max()/ct.groupby(level=0).sum()
def engineer_features(df,entity="publisher",min_events=8):
    if entity not in ("publisher","campaign","slot_id"):raise ValueError("Unknown entity")
    g=df.groupby(entity,dropna=False)
    out=g.agg(events=("click","size"),clicks=("click","sum"),unique_ips=("ip_key","nunique"),device_diversity=("device","nunique"),campaign_diversity=("campaign","nunique"),slot_diversity=("slot_id","nunique"),avg_price=("price","mean"))
    out["ctr"]=out.clicks/out.events
    out["log_events"]=np.log1p(out.events)
    out["events_per_ip"]=out.events/out.unique_ips.clip(lower=1)
    out["ip_concentration"]=maxshare(df,entity,"ip_key")
    out["hour_concentration"]=maxshare(df,entity,"hour_bucket")
    ips=df[[entity,"ip_key"]].drop_duplicates()
    degrees=ips.groupby("ip_key")[entity].nunique()
    out["cross_source_ip_share"]=df.ip_key.map(degrees).gt(1).groupby(df[entity]).mean()
    if df.event_time.notna().any():
        tmp=df[[entity,"event_time"]].dropna().copy()
        tmp["bucket"]=pd.to_datetime(tmp.event_time,utc=True).dt.floor("10min")
        out["burst_10m_share"]=maxshare(tmp,entity,"bucket")
    else:out["burst_10m_share"]=np.nan
    out["publisher_label"]=g.bot_label.max()
    out["injected_events"]=g.is_injected.sum()
    out=out[out.events>=min_events].copy()
    return out.reset_index().rename(columns={entity:"entity_id"})
def numeric_matrix(df):
    cols=[c for c in FEATURES if c in df and df[c].notna().any() and df[c].nunique(dropna=True)>1]
    if not cols:raise ValueError("No varying numeric features")
    x=df[cols].replace([np.inf,-np.inf],np.nan)
    return x.fillna(x.median()).fillna(0),cols
