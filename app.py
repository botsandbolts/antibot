"""AntiBot Lab — Streamlit research application."""
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from fraudlab.data import make_demo,normalize,inject_bots,load_ipinyou
from fraudlab.features import engineer_features
from fraudlab.models import UNSUPERVISED,anomaly_scores,benchmark
from fraudlab import charts

st.set_page_config(page_title="AntiBot Lab | Advertising Intelligence",page_icon="◈",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@400;600;700&display=swap');
html,body{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.03em}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse 650px 370px at 85% 2%,rgba(105,115,190,.17),transparent),#08101E}
[data-testid="stSidebar"]{background:#0B1627;border-right:1px solid #213651}
[data-testid="stHeader"]{background:rgba(8,16,30,.7)}
[data-testid="stMetric"]{background:linear-gradient(120deg,#14283B,#121F37);border:1px solid #28415A;border-radius:15px;padding:18px}
.hero{background:linear-gradient(115deg,#143343,#152341 70%,#232344);border:1px solid #32506A;border-radius:23px;padding:30px;margin-bottom:20px}
.hero h1{font-size:40px;color:#F6FBFF;margin:4px 0 9px}
.hero p{color:#C0D2E1;font-size:15px}
.eyebrow{color:#65F4C4;font-size:11px;letter-spacing:.21em;font-weight:700}
.brand{font-family:'Space Grotesk',sans-serif;font-size:28px;color:#F0F8FF;font-weight:700}
.brand span{color:#65F4C4}
.chip{display:inline-block;padding:6px 13px;border:1px solid #315852;border-radius:99px;background:#11342f;color:#92F8DC;margin:6px 5px 13px 0;font-size:12px}
</style>""",unsafe_allow_html=True)
def hero(eyebrow,title,description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>',unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def demo(seed):return make_demo(seed=seed)
@st.cache_data(show_spinner=False)
def score_rank(f,method,budget):
    out=f.copy();out["score"]=anomaly_scores(f,method,contamination=budget)
    out=out.sort_values("score",ascending=False).reset_index(drop=True)
    out["anomaly_percentile"]=100*(1-(out.index.to_numpy())/max(1,len(out)))
    return out

with st.sidebar:
    st.markdown('<div class="brand">◈ Anti<span>Bot</span> Lab</div><small style="color:#8BA8BE;letter-spacing:.16em">ADVERTISING INTELLIGENCE</small>',unsafe_allow_html=True)
    st.divider()
    page=st.radio("RESEARCH WORKSPACES",["◈  Command center","⌁  Traffic explorer","◎  Anomaly studio","⟡  Graph intelligence","▥  Model arena","⚡  Red-team simulator","ⓘ  Research notes"],label_visibility="collapsed")
    st.divider()
    data_mode=st.selectbox("Dataset",["Synthetic sandbox","iPinYou x1 (real, unlabeled)","Upload advertising CSV"])
    seed=st.number_input("Reproducibility seed",0,999999,42)
    if data_mode=="Synthetic sandbox":
        base=demo(seed); origin="Synthetic, fully known event labels"
    elif data_mode=="iPinYou x1 (real, unlabeled)":
        scan=st.selectbox("Scan training records",["500,000","1,000,000","Full archive"])
        keep=st.slider("Retain events",10000,100000,60000,10000)
        if st.button("↓  Download iPinYou",type="primary",use_container_width=True):
            try:
                with st.spinner("Downloading archive and sampling traffic …"):
                    d,count=load_ipinyou(keep,0 if scan=="Full archive" else int(scan.replace(",","")),seed)
                st.session_state["real_data"]=d;st.session_state["scan_count"]=count
            except Exception as e:st.error(f"Download failed: {e}")
        if "real_data" not in st.session_state:
            st.info("Real dataset not bundled. Press Download to fetch the non-commercial iPinYou benchmark.")
            st.stop()
        base=st.session_state.real_data;origin=f"Real iPinYou; {st.session_state.scan_count:,} records scanned. No verified bot labels."
    else:
        upload=st.file_uploader("Upload CSV",type=["csv"])
        if upload is None:st.info("Upload a CSV to continue.");st.stop()
        raw=pd.read_csv(upload,nrows=120000)
        label=st.selectbox("Optional label column",["None"]+raw.columns.tolist())
        base=normalize(raw,label_column=None if label=="None" else label)
        origin="Uploaded data, labels are user-declared and unaudited"
    entity=st.selectbox("Analysis entity",["publisher","slot_id","campaign"])
    min_events=st.slider("Minimum events per entity",3,100,8)
    budget=st.slider("Anomaly review fraction",.01,.30,.08,.01,help="This is not a fraud prevalence estimate.")
    if "injected_data" in st.session_state and st.session_state.get("inject_origin")==data_mode:
        show=st.toggle("Include latest simulated attack",True)
        df=st.session_state.injected_data if show else base
    else:df=base
    st.caption("Research only · rankings ≠ fraud verdicts")

features=engineer_features(df,entity,min_events)
has_truth=bool(len(features) and features.publisher_label.notna().all() and features.publisher_label.nunique()==2)
st.markdown(f'<span class="chip">{"SYNTHETIC GROUND TRUTH" if data_mode=="Synthetic sandbox" else "REAL TRAFFIC · BOT STATUS UNKNOWN"}</span><span class="chip">UNIT · {entity.upper()}</span>',unsafe_allow_html=True)

if page.startswith("◈"):
    hero("01 / COMMAND CENTER","Signals, not suspicions.","A sharp view of advertising behavior, traffic concentration and sources worth investigating.")
    cols=st.columns(4)
    for c,(label,val) in zip(cols,[("Events",f"{len(df):,}"),("Traffic sources",f"{len(features):,}"),("CTR",f"{100*df.click.mean():.2f}%"),("Injected events",f"{int(df.is_injected.sum()):,}")]):c.metric(label,val)
    st.info("Known bot ground truth exists only for explicitly synthetic events. Real network events without independent labels remain unknown.")
    l,r=st.columns([1.35,1])
    with l:st.subheader("Traffic pulse");st.plotly_chart(charts.traffic_curve(df),use_container_width=True)
    with r:st.subheader("Traffic concentration");st.plotly_chart(charts.source_volume(features),use_container_width=True)
    st.caption(origin)
elif page.startswith("⌁"):
    hero("02 / DATA EXPLORER","Behind every impression.","Study traffic sources, click behavior, campaign volume, and encoded device/IP diversity.")
    a,b,c=st.columns(3)
    a.metric("Source identifiers",f"{df.publisher.nunique():,}")
    b.metric("Ad slots",f"{df.slot_id.nunique():,}")
    c.metric("Encoded IP keys",f"{df.ip_key.nunique():,}")
    t1,t2,t3=st.tabs(["Traffic volume","Campaign analysis","Raw events"])
    with t1:st.plotly_chart(charts.source_volume(features),use_container_width=True)
    with t2:
        perf=df.groupby("campaign").agg(Events=("click","size"),Clicks=("click","sum"),Mean_price=("price","mean"))
        perf["CTR"]=perf.Clicks/perf.Events
        st.dataframe(perf.sort_values("Events",ascending=False).head(40),use_container_width=True)
    with t3:
        st.dataframe(df.head(3000),use_container_width=True,height=480)
        st.download_button("Export sample CSV",df.head(5000).to_csv(index=False),"sample.csv","text/csv")
elif page.startswith("◎"):
    hero("03 / ANOMALY STUDIO","The interesting outliers.","Explore rare publisher patterns using multiple unsupervised detectors. Ranks are not posterior fraud probabilities.")
    if len(features)<9:st.warning("Too few groups. Lower minimum events.")
    else:
        method=st.selectbox("Detection method",UNSUPERVISED)
        ranked=score_rank(features,method,budget)
        a,b=st.columns([1.3,1])
        with a:st.subheader("Behavior landscape");st.plotly_chart(charts.risk_map(ranked),use_container_width=True)
        with b:st.subheader("Highest anomaly scores");st.plotly_chart(charts.score_ranking(ranked),use_container_width=True)
        st.dataframe(ranked[["entity_id","events","ctr","unique_ips","ip_concentration","score","anomaly_percentile"]].head(80),use_container_width=True)
        st.download_button("Export ranked anomalies",ranked.to_csv(index=False),"anomaly_ranking.csv","text/csv")
elif page.startswith("⟡"):
    hero("04 / GRAPH INTELLIGENCE","Connections tell stories.","Explore relationships between publishers, advertiser placements and encoded IP keys.")
    if features.empty:st.warning("Not enough source data.")
    else:
        top=features.nlargest(15,"events").entity_id.tolist()
        selected=st.multiselect("Source groups",top,default=top[:8])
        fig,nodes,edges=charts.ip_graph(df,entity,selected)
        a,b=st.columns(2);a.metric("Nodes",nodes);b.metric("Co-occurrence edges",edges)
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Shared IP keys may reflect legitimate network gateways. This is a graph visualization, not a trained GNN.")
elif page.startswith("▥"):
    hero("05 / MODEL ARENA","Let the evidence compete.","Compare unsupervised baselines and, where trustworthy labels exist, supervised models on held-out traffic sources.")
    if len(features)<20:st.warning("At least 20 groups required. Reduce minimum events.")
    else:
        with st.spinner("Evaluating models …"):
            summary,details=benchmark(features,contamination=budget)
        if not has_truth:st.warning("No complete source labels: accuracy, PR-AUC and fraud recall intentionally unavailable.")
        else:st.info("Synthetic or user-supplied labels only. No claim of verified real-world bot status.")
        st.dataframe(summary,use_container_width=True)
        valid=summary.dropna(subset=["PR-AUC"]) if "PR-AUC" in summary else pd.DataFrame()
        if len(valid):st.plotly_chart(charts.polish(px.bar(valid.sort_values("PR-AUC"),x="PR-AUC",y="Model",orientation="h",color="Track")),use_container_width=True)
        st.download_button("Export leaderboard",summary.to_csv(index=False),"model_leaderboard.csv","text/csv")
        st.caption("Source-disjoint splits; feature engineering uses global graph overlap (retrospective/transductive). Do not mistake it for prospective deployment performance.")
elif page.startswith("⚡"):
    hero("06 / RED-TEAM SIMULATOR","Attack the assumptions.","Inject controlled adversarial ad traffic into the selected baseline without assigning fabricated labels to real visitors.")
    attack=st.selectbox("Attack pattern",["Click storm","Device rotation","Low and slow"])
    ratio=st.slider("Injected share relative to baseline",.01,.3,.08,.01)
    if st.button("Inject simulated activity",type="primary"):
        st.session_state.injected_data=inject_bots(base,ratio,attack,seed)
        st.session_state.inject_origin=data_mode
        st.success(f"Injected {int(st.session_state.injected_data.is_injected.sum()) - int(base.is_injected.sum()):,} additional synthetic events. Explore results in Model arena.")
    st.write("Attack patterns alter IP reuse, device diversity and click behavior. Injected positives have ground truth; existing unknown real events stay unknown.")
elif page.startswith("ⓘ"):
    hero("07 / RESEARCH NOTES","An honest benchmark.","Maintain a strict boundary between detected outliers, publisher fraud labels, and confirmed automated clients.")
    st.markdown("""### Methodology
- **Synthetic demo:** full simulated binary labels; no authentic people or advertisers.
- **Real iPinYou:** real ad-event patterns with no verified bot labels; publisher/domain is a proxy.
- **Model comparison:** publisher/slot/campaign-disjoint validation, not chronological validation.
- **Graph overlap:** transductive cross-source co-occurrence; not a graph neural network.
- **Interpretation:** abnormal clicks or shared IPs do not prove bots or criminal activity.
### Data references
[iPinYou benchmark](https://huggingface.co/datasets/reczoo/iPinYou_x1) · [Competition](https://contest.ipinyou.com/)
The iPinYou material is subject to third-party non-commercial research rights.
""")
st.divider()
st.caption("AntiBot Lab · Advertising intelligence research · Built with Streamlit, Plotly and scikit-learn")
