"""Synthetic ad events and real-data normalization; unknown labels remain unknown."""
import numpy as np
import pandas as pd

def normalize(raw, label_column=None, real_ipinyou=False):
    raw=raw.copy()
    names={str(c).lower():c for c in raw.columns}
    def col(keys, default):
        for k in keys:
            if k.lower() in names:return raw[names[k.lower()]]
        return pd.Series([default]*len(raw),index=raw.index)
    out=pd.DataFrame(index=raw.index)
    columns={
      "click":(["click","clicked"],0),
      "publisher":(["publisher","publisherid","domain","site_id"],"unknown"),
      "ip_key":(["ip_key","IP","numericip"],"unknown"),
      "device":(["device","useragent","deviceua"],"unknown"),
      "campaign":(["campaign","campaignid","advertiser"],"unknown"),
      "slot_id":(["slot_id","slotid","adslot"],"unknown"),
      "price":(["price","slotprice","payingprice"],0),
      "hour_bucket":(["hour_bucket","hour"],"unknown"),
      "event_time":(["event_time","timestamp","clicktime"],pd.NaT)}
    for key,(aliases,default) in columns.items():out[key]=col(aliases,default)
    out["click"]=pd.to_numeric(out["click"],errors="coerce").fillna(0).gt(0).astype(int)
    out["price"]=pd.to_numeric(out["price"],errors="coerce").fillna(0).clip(lower=0)
    for key in ("publisher","ip_key","device","campaign","slot_id","hour_bucket"):
        out[key]=out[key].astype("string").fillna("unknown")
    if out["publisher"].eq("unknown").all():out["publisher"]="slot:"+out["slot_id"]
    out["event_time"]=pd.NaT if real_ipinyou else pd.to_datetime(out["event_time"],utc=True,errors="coerce")
    if real_ipinyou:out["hour_bucket"]="encoded:"+out["hour_bucket"]
    out["bot_label"]=np.nan
    if label_column and label_column in raw:
        v=pd.to_numeric(raw[label_column],errors="coerce")
        out["bot_label"]=v.where(v.isin([0,1]),np.nan)
    out["is_injected"]=False
    return out.reset_index(drop=True)

def make_demo(n=24000, seed=42):
    rng=np.random.default_rng(seed)
    sources=220
    probs=rng.pareto(2,sources)+.4; probs/=probs.sum()
    pub=rng.choice(sources,n,p=probs)
    bad_sources=rng.choice(sources,32,replace=False)
    bot=np.isin(pub,bad_sources)&(rng.random(n)<.65)
    ip=rng.integers(0,6000,n)
    ip[bot]=rng.integers(100000,100040,bot.sum())
    dev=rng.integers(0,14,n);dev[bot]=rng.integers(0,2,bot.sum())
    clicks=rng.random(n)<np.where(bot,.42,.032)
    sec=rng.integers(0,86400*10,n)
    dt=pd.Timestamp("2026-01-01",tz="UTC")+pd.to_timedelta(sec,unit="s")
    return pd.DataFrame({"click":clicks.astype(int),"publisher":[f"source-{x:03d}" for x in pub],
        "ip_key":[f"ip-{x}" for x in ip],"device":[f"device-{x}" for x in dev],
        "campaign":[f"campaign-{x}" for x in rng.integers(0,25,n)],
        "slot_id":[f"slot-{x}" for x in rng.integers(0,80,n)],
        "price":rng.gamma(2,15,n).round(2),"hour_bucket":dt.hour.astype(str),
        "event_time":dt,"bot_label":bot.astype(int),"is_injected":bot})

def inject_bots(base, ratio=.08, attack="Click storm", seed=43):
    rng=np.random.default_rng(seed); count=max(50,int(len(base)*ratio))
    attack_rows=base.iloc[rng.integers(0,len(base),count)].copy().reset_index(drop=True)
    targets=base.publisher.value_counts().head(20).index.to_numpy()
    attack_rows["publisher"]=rng.choice(targets,size=count)
    attack_rows["bot_label"]=1;attack_rows["is_injected"]=True
    attack_rows["ip_key"]=[f"SIM-{i% (8 if attack=='Click storm' else 90)}" for i in range(count)]
    attack_rows["device"]=[f"SIM-UA-{i% (1 if attack=='Click storm' else 80)}" for i in range(count)]
    attack_rows["click"]=(rng.random(count)< (0.85 if attack=="Click storm" else .08 if attack=="Low and slow" else .3)).astype(int)
    if base.event_time.notna().any():
        first=base.event_time.min(); last=base.event_time.max()
        span=(last-first).total_seconds()
        times=rng.uniform(span*.8,span*.8+max(1,span*.003),count) if attack=="Click storm" else rng.uniform(0,span,count)
        attack_rows["event_time"]=first+pd.to_timedelta(times,unit="s")
        attack_rows["hour_bucket"]=pd.to_datetime(attack_rows.event_time).dt.hour.astype(str)
    return pd.concat([base,attack_rows],ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)

def load_ipinyou(sample_size=60000,max_scan=1000000,seed=42):
    from huggingface_hub import hf_hub_download
    import zipfile
    path=hf_hub_download(repo_id="reczoo/iPinYou_x1",filename="iPinYou_x1.zip",repo_type="dataset")
    rng=np.random.default_rng(seed); held=[];total=0
    with zipfile.ZipFile(path) as z:
        files=[x for x in z.namelist() if x.lower().endswith("train.csv")]
        if not files:files=[x for x in z.namelist() if x.lower().endswith(".csv")]
        if not files:raise ValueError("No CSV in archive")
        with z.open(files[0]) as fh:
            for chunk in pd.read_csv(fh,chunksize=60000):
                if max_scan:chunk=chunk.iloc[:max(0,max_scan-total)]
                if chunk.empty:break
                total+=len(chunk)
                # Deterministic priority-based bounded reservoir of chunks.
                portion=min(len(chunk),max(100,int(sample_size*len(chunk)/max(total,len(chunk)))))
                held.append(chunk.sample(n=portion,random_state=seed+len(held)))
                if max_scan and total>=max_scan:break
    data=pd.concat(held,ignore_index=True)
    if len(data)>sample_size:data=data.sample(n=sample_size,random_state=seed)
    return normalize(data,real_ipinyou=True),total
