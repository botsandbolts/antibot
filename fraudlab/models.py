"""Reproducible source-disjoint anomaly and supervised benchmarks."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import train_test_split
from .features import numeric_matrix
UNSUPERVISED=["Isolation Forest","ECOD tail score","Local Outlier Factor","One-Class SVM","IP overlap"]
SUPERVISED=["Logistic Regression","Random Forest","HistGradientBoosting"]
def ecod(train,test):
    scores=[]
    for j in range(train.shape[1]):
        s=np.sort(train[:,j])
        pct=np.searchsorted(s,test[:,j],side="right")/len(s)
        scores.append(-np.log(np.clip(np.minimum(pct,1-pct),1/(len(s)+1),1)))
    return np.stack(scores,axis=1).max(axis=1)
def graph_score(f):
    x=f[["cross_source_ip_share","ip_concentration"]].rank(pct=True).fillna(.5)
    return (.6*x.iloc[:,0]+.4*x.iloc[:,1]).to_numpy()
def fit_score(method,xt,xv,yt=None,seed=42,contamination=.08):
    if method=="Isolation Forest":
        m=IsolationForest(n_estimators=160,contamination=contamination,random_state=seed,n_jobs=-1)
        return -m.fit(xt).score_samples(xv)
    if method=="ECOD tail score":return ecod(xt,xv)
    if method=="Local Outlier Factor":
        m=LocalOutlierFactor(n_neighbors=min(20,len(xt)-1),novelty=True)
        return -m.fit(xt).score_samples(xv)
    if method=="One-Class SVM":return -OneClassSVM(nu=contamination).fit(xt).score_samples(xv)
    if method=="Logistic Regression":m=LogisticRegression(max_iter=800,class_weight="balanced")
    elif method=="Random Forest":m=RandomForestClassifier(n_estimators=180,min_samples_leaf=2,class_weight="balanced_subsample",random_state=seed,n_jobs=-1)
    elif method=="HistGradientBoosting":m=HistGradientBoostingClassifier(max_iter=120,l2_regularization=3,random_state=seed)
    else:raise ValueError(method)
    return m.fit(xt,yt).predict_proba(xv)[:,1]
def anomaly_scores(features,method,contamination=.08):
    if method=="IP overlap":return graph_score(features)
    x,_=numeric_matrix(features);z=RobustScaler().fit_transform(x)
    if method=="Local Outlier Factor":
        m=LocalOutlierFactor(n_neighbors=min(20,len(z)-1))
        return -m.fit_predict(z)*0+(-m.negative_outlier_factor_)
    return fit_score(method,z,z,seed=42,contamination=contamination)
def benchmark(features,contamination=.08,review_budget=.1,seed=42):
    if len(features)<20:raise ValueError("Need >=20 eligible groups")
    x,_=numeric_matrix(features)
    y=features.publisher_label
    labeled=y.notna().all() and y.nunique()==2 and y.value_counts().min()>=3
    ids=np.arange(len(features))
    train,test=train_test_split(ids,test_size=.3,random_state=seed,stratify=y.astype(int) if labeled else None)
    scaler=RobustScaler().fit(x.iloc[train])
    xt=scaler.transform(x.iloc[train]);xv=scaler.transform(x.iloc[test])
    yt=y.iloc[train].astype(int).to_numpy() if labeled else None
    yv=y.iloc[test].astype(int).to_numpy() if labeled else None
    rows=[];details=[]
    for name in UNSUPERVISED+(SUPERVISED if labeled else []):
        try:
            s=graph_score(features.iloc[test]) if name=="IP overlap" else fit_score(name,xt,xv,yt,seed,contamination)
            k=max(1,int(np.ceil(len(test)*review_budget)))
            best=np.argsort(-s)[:k]
            item={"Model":name,"Track":"Supervised" if name in SUPERVISED else "Unsupervised","PR-AUC":np.nan,"ROC-AUC":np.nan,"Precision@K":np.nan,"Recall@K":np.nan}
            if labeled and len(np.unique(yv))==2:
                item.update({"PR-AUC":average_precision_score(yv,s),"ROC-AUC":roc_auc_score(yv,s),"Precision@K":yv[best].mean(),"Recall@K":yv[best].sum()/max(1,yv.sum())})
            rows.append(item)
            details.extend({"Model":name,"entity_id":str(features.iloc[test[j]].entity_id),"score":float(s[j]),"target":int(yv[j]) if labeled else np.nan} for j in range(len(test)))
        except Exception as e:rows.append({"Model":name,"Track":"Failed","Error":str(e)})
    return pd.DataFrame(rows),pd.DataFrame(details)
