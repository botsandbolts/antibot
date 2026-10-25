import numpy as np
import pandas as pd
from fraudlab.data import make_demo,normalize,inject_bots
from fraudlab.features import engineer_features
from fraudlab.models import anomaly_scores,benchmark,UNSUPERVISED
from fraudlab.charts import traffic_curve,ip_graph

def test_synthetic_reproducible():
    a=make_demo(n=3000,seed=7);b=make_demo(n=3000,seed=7)
    assert a.equals(b)
    assert a.bot_label.notna().all()
    assert set(a.bot_label.unique())=={0,1}

def test_real_labels_unknown():
    d=normalize(pd.DataFrame({"IP":[1,2],"domain":["a","b"],"click":[0,1],"hour":[3,4]}),real_ipinyou=True)
    assert d.bot_label.isna().all()
    assert d.event_time.isna().all()
    assert d.hour_bucket.iloc[0]=="encoded:3"

def test_injected_only_labels():
    raw=make_demo(3000)[["click","publisher","ip_key","device","campaign","slot_id","price"]]
    base=normalize(raw)
    attacked=inject_bots(base,ratio=.10)
    assert attacked.loc[~attacked.is_injected,"bot_label"].isna().all()
    assert attacked.loc[attacked.is_injected,"bot_label"].eq(1).all()

def test_anomalies():
    d=make_demo(n=10000)
    f=engineer_features(d,min_events=5)
    for name in UNSUPERVISED:
        z=anomaly_scores(f,name)
        assert len(z)==len(f)
        assert np.isfinite(z).all()

def test_supervised_benchmark():
    f=engineer_features(make_demo(n=12000),min_events=5)
    summary,detail=benchmark(f)
    assert len(detail)>0
    assert "Random Forest" in summary.Model.values
    assert summary["PR-AUC"].dropna().between(0,1).all()

def test_unlabeled_no_fake_auc():
    d=normalize(make_demo(12000)[["click","publisher","ip_key","device","campaign","slot_id","price"]])
    f=engineer_features(d,min_events=5)
    summary,detail=benchmark(f)
    assert summary["PR-AUC"].isna().all()
    assert "Random Forest" not in summary.Model.values
    assert detail.target.isna().all()

def test_charts():
    d=make_demo(4000)
    assert len(traffic_curve(d).data)>0
    fig,nodes,edges=ip_graph(d)
    assert nodes>0 and edges>0
