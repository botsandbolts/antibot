"""Interactive plots with a restrained research aesthetic."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
MINT="#65F4C4";LAVENDER="#AFA9FF"
def polish(fig,height=385):
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#D9E8FA"),height=height,margin=dict(l=12,r=12,t=34,b=24),colorway=[MINT,LAVENDER,"#FFAD8A"])
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor="rgba(122,159,187,.13)",zeroline=False)
    return fig
def traffic_curve(df):
    if df.event_time.notna().any():
        temp=df.assign(bucket=pd.to_datetime(df.event_time,utc=True).dt.floor("6h")).groupby("bucket").size().reset_index(name="events");x="bucket"
    else:
        temp=df.groupby("hour_bucket").size().reset_index(name="events");x="hour_bucket"
    return polish(px.area(temp,x=x,y="events",color_discrete_sequence=[MINT]),360)
def source_volume(f):
    data=f.nlargest(16,"events").sort_values("events")
    return polish(px.bar(data,x="events",y="entity_id",orientation="h",color_discrete_sequence=[MINT]),360)
def risk_map(f):
    return polish(px.scatter(f,x="log_events",y="ctr",size="events",color="score",hover_name="entity_id",color_continuous_scale=["#1E5772",LAVENDER,"#FFAD8A"]),405)
def score_ranking(f):
    return polish(px.bar(f.nlargest(18,"score").sort_values("score"),x="score",y="entity_id",orientation="h",color_discrete_sequence=[LAVENDER]),405)
def ip_graph(df,entity="publisher",selected=None):
    if selected is None:selected=df[entity].value_counts().head(12).index.tolist()
    pairs=df[df[entity].isin(selected)][[entity,"ip_key"]].drop_duplicates().head(300)
    frequent=pairs.ip_key.value_counts().head(35).index
    pairs=pairs[pairs.ip_key.isin(frequent)]
    g=nx.Graph()
    for _,r in pairs.iterrows():g.add_edge("S:"+str(r[entity]),"IP:"+str(r.ip_key))
    if not len(g):return polish(go.Figure(),400),0,0
    positions=nx.spring_layout(g,seed=42)
    edges_x=[];edges_y=[]
    for a,b in g.edges():
        edges_x.extend([positions[a][0],positions[b][0],None]);edges_y.extend([positions[a][1],positions[b][1],None])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=edges_x,y=edges_y,mode="lines",line=dict(color="rgba(140,164,200,.22)",width=1),hoverinfo="skip"))
    for prefix,color in [("S:",MINT),("IP:",LAVENDER)]:
        nodes=[n for n in g if n.startswith(prefix)]
        fig.add_trace(go.Scatter(x=[positions[n][0] for n in nodes],y=[positions[n][1] for n in nodes],text=nodes,mode="markers",marker=dict(size=10 if prefix=="S:" else 6,color=color),hovertemplate="%{text}<extra></extra>",name="Traffic source" if prefix=="S:" else "IP key"))
    fig.update_xaxes(visible=False);fig.update_yaxes(visible=False)
    return polish(fig,475),len(g),g.number_of_edges()
