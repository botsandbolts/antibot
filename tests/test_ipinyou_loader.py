"""Offline fixture for the exact published iPinYou x1 column names.
This exercises the real ZIP ingestion pathway without pretending to download the full archive.
"""
import zipfile
import numpy as np
import pandas as pd
from fraudlab.data import load_ipinyou

def test_real_ipinyou_schema_and_zip_loader(tmp_path,monkeypatch):
    rows=250
    raw=pd.DataFrame({
        "click":np.arange(rows)%2,
        "weekday":[1+(i%7) for i in range(rows)],
        "hour":[9+(i%24) for i in range(rows)],
        "IP":[33+(i%12) for i in range(rows)],
        "domain":[705000+(i%7) for i in range(rows)],
        "slotid":[757000+(i%18) for i in range(rows)],
        "advertiser":[938000+(i%3) for i in range(rows)],
        "useragent":[938100+(i%9) for i in range(rows)],
        "slotprice":np.arange(rows)%50,
    })
    path=tmp_path/"iPinYou_x1.zip"
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as arc:
        arc.writestr("train.csv",raw.to_csv(index=False))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub,"hf_hub_download",lambda **kwargs: str(path))
    sample,scanned=load_ipinyou(sample_size=70,max_scan=200,seed=17)
    assert scanned==200
    assert len(sample)==70
    assert sample.publisher.nunique()>1
    assert sample.ip_key.nunique()>1
    assert sample.campaign.nunique()>1
    assert sample.slot_id.nunique()>1
    assert sample.bot_label.isna().all()
    assert sample.event_time.isna().all()
    assert sample.hour_bucket.str.startswith("encoded:").all()
    again,_=load_ipinyou(sample_size=70,max_scan=200,seed=17)
    assert sample.equals(again)
