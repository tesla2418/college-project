import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# Page setup
st.set_page_config(page_title="Crypto Fraud Detector", page_icon="🔍", layout="wide")

st.title("🔍 Crypto Flow & Fraud Analytics")
st.markdown("""
Upload apni transaction database (CSV files). System batayega ki kaun si transactions risky (fraud) ho sakti hain 
aur uska graphical analysis bhi dikhayega.
""")

# ============================================================
# SIDEBAR: File Upload
# ============================================================
st.sidebar.header("📁 Upload Database Files")
edges_file = st.sidebar.file_uploader(
    "Upload Edgelist CSV", 
    type=["csv"], 
    help="Isme columns hone chahiye: txId1, txId2"
)
classes_file = st.sidebar.file_uploader(
    "Upload Classes CSV", 
    type=["csv"], 
    help="Isme columns hone chahiye: txId, class"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Column Requirements:**
- Edgelist: `txId1`, `txId2`
- Classes: `txId`, `class` (values: 1=illicit, 2=licit, unknown)
""")

# ============================================================
# CORE DETECTION LOGIC
# ============================================================
def run_detection(edges_df, classes_df):
    # Label dictionary: ID -> label
    label = dict(zip(classes_df["txId"], classes_df["class"].astype(str)))
    illicit_set = set(tx for tx, c in label.items() if c == "1")
    
    # Graph banana (Adjacency List)
    graph = defaultdict(list)      # Directed: a -> b
    undirected = defaultdict(list) # Undirected: a <-> b
    for a, b in zip(edges_df["txId1"], edges_df["txId2"]):
        graph[a].append(b)
        undirected[a].append(b)
        undirected[b].append(a)
    
    # BFS: Fraud se 2 hop tak ke andar kaun hai
    near = {}
    queue = deque()
    for tx in illicit_set:
        near[tx] = 0
        queue.append(tx)
    
    while queue:
        current = queue.popleft()
        if near[current] >= 2:
            continue
        for neighbor in undirected[current]:
            if neighbor not in near:
                near[neighbor] = near[current] + 1
                queue.append(neighbor)
    
    # Union-Find: Connected components (groups)
    parent = {}
    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    
    for tx in undirected:
        parent[tx] = tx
    for a, b in zip(edges_df["txId1"], edges_df["txId2"]):
        union(a, b)
    
    groups = defaultdict(list)
    for tx in undirected:
        groups[find(tx)].append(tx)
    
    # Cycle Detection (DFS)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {tx: WHITE for tx in undirected}
    has_cycle = False
    for start in undirected:
        if color[start] != WHITE:
            continue
        stack = [(start, 0)]
        color[start] = GRAY
        while stack and not has_cycle:
            node, phase = stack.pop()
            if phase == 0:
                stack.append((node, 1))
                for neighbor in graph[node]:
                    if color[neighbor] == GRAY:
                        has_cycle = True
                        break
                    if color[neighbor] == WHITE:
                        color[neighbor] = GRAY
                        stack.append((neighbor, 0))
            else:
                color[node] = BLACK
        if has_cycle:
            break
    
    # Flagging Rules
    flagged = []
    for tx in undirected:
        lbl = label.get(tx, "unknown")
        if lbl == "1":
            flagged.append({
                "txId": tx,
                "reason": "ALREADY ILLICIT",
                "hop": 0,
                "risk": "HIGH"
            })
        elif lbl == "unknown" and tx in near:
            hops = near[tx]
            if hops == 1:
                flagged.append({
                    "txId": tx,
                    "reason": "HIGH - 1 hop from illicit",
                    "hop": hops,
                    "risk": "HIGH"
                })
            elif hops == 2:
                flagged.append({
                    "txId": tx,
                    "reason": "MEDIUM - 2 hops from illicit",
                    "hop": hops,
                    "risk": "MEDIUM"
                })
    
    flagged_df = pd.DataFrame(flagged)
    
    # Evaluation (sirf jin ka label pata hai un par)
    tp = fp = total_illicit = 0
    flagged_set = set(flagged_df["txId"]) if not flagged_df.empty else set()
    
    for tx, lbl in label.items():
        if lbl not in ("1", "2"):
            continue
        if tx in flagged_set:
            if lbl == "1":
                tp += 1
            else:
                fp += 1
        if lbl == "1":
            total_illicit += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_illicit if total_illicit > 0 else 0
    
    return {
        "label": label,
        "illicit_set": illicit_set,
        "graph": graph,
        "undirected": undirected,
        "near": near,
        "num_groups": len(groups),
        "largest_group": max(len(v) for v in groups.values()) if groups else 0,
        "has_cycle": has_cycle,
        "flagged_df": flagged_df,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "total_illicit": total_illicit,
        "total_nodes": len(undirected),
        "total_edges": len(edges_df)
    }

# ============================================================
# MAIN APP LOGIC
# ============================================================
if edges_file and classes_file:
    try:
        edges_df = pd.read_csv(edges_file)
        classes_df = pd.read_csv(classes_file)
        
        # Column check
        if not {"txId1", "txId2"}.issubset(edges_df.columns):
            st.error("❌ Edgelist file mein columns hone chahiye: **txId1, txId2**")
            st.stop()
        if not {"txId", "class"}.issubset(classes_df.columns):
            st.error("❌ Classes file mein columns hone chahiye: **txId, class**")
            st.stop()
        
        # Run detection
        with st.spinner("Analysis chal raha hai, thoda intezaar karein..."):
            result = run_detection(edges_df, classes_df)
        
        st.success("✅ Analysis complete!")
        
        # ============================================================
        # METRICS ROW
        # ============================================================
        st.subheader("📊 Summary Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Transactions", f"{result['total_nodes']:,}")
        col2.metric("Total Connections", f"{result['total_edges']:,}")
        col3.metric("Flagged Risky", f"{len(result['flagged_df']):,}")
        col4.metric("Connected Groups", result["num_groups"])
        col5.metric("Cycle Detected", "✅ Yes" if result["has_cycle"] else "❌ No")
        
        # ============================================================
        # CHARTS
        # ============================================================
        st.subheader("📈 Graphical Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        # Risk distribution bar chart
        with chart_col1:
            st.write("**Flagged Transactions by Risk Level**")
            if not result["flagged_df"].empty:
                risk_counts = result["flagged_df"]["risk"].value_counts().reset_index()
                risk_counts.columns = ["Risk", "Count"]
                fig1 = px.bar(risk_counts, x="Risk", y="Count", color="Risk",
                             color_discrete_map={"HIGH": "#ff4b4b", "MEDIUM": "#ffa500"})
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("Koi transaction flagged nahi hui.")
        
        # Label distribution pie chart
        with chart_col2:
            st.write("**Transaction Label Distribution**")
            label_series = pd.Series(result["label"]).value_counts().reset_index()
            label_series.columns = ["Label", "Count"]
            label_map = {"1": "Illicit", "2": "Licit", "unknown": "Unknown"}
            label_series["Label"] = label_series["Label"].map(label_map).fillna(label_series["Label"])
            fig2 = px.pie(label_series, names="Label", values="Count", 
                         color_discrete_map={"Illicit": "#ff4b4b", "Licit": "#4bff4b", "Unknown": "#cccccc"})
            st.plotly_chart(fig2, use_container_width=True)
        
        # ============================================================
        # EVALUATION
        # ============================================================
        st.subheader("🎯 Model Evaluation")
        eval_col1, eval_col2, eval_col3 = st.columns(3)
        eval_col1.metric("Precision", f"{result['precision']:.2%}")
        eval_col2.metric("Recall", f"{result['recall']:.2%}")
        eval_col3.metric("True Positives", result["tp"])
        st.write(f"False Positives: {result['fp']} | Total Actual Illicit: {result['total_illicit']}")
        
        # ============================================================
        # FLAGGED TABLE
        # ============================================================
        st.subheader("🚩 Flagged Transactions")
        if not result["flagged_df"].empty:
            st.dataframe(result["flagged_df"], use_container_width=True, height=400)
            
            # Download button
            csv = result["flagged_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Flagged Transactions CSV",
                data=csv,
                file_name="flagged_transactions.csv",
                mime="text/csv"
            )
        else:
            st.warning("Koi risky transaction nahi mili is database mein.")
        
        # ============================================================
        # NETWORK GRAPH VISUALIZATION
        # ============================================================
        st.subheader("🕸️ Network Graph (Sample View)")
        st.info("Poora graph bahut bada hona hai, isliye sirf flagged nodes ke aas-paas ka chhota sample dikhaya ja raha hai.")
        
        flagged_sample = result["flagged_df"]["txId"].head(30).tolist() if not result["flagged_df"].empty else []
        nodes_to_draw = set(flagged_sample)
        
        # Add neighbours of flagged nodes
        for tx in flagged_sample:
            if tx in result["undirected"]:
                nodes_to_draw.add(tx)
                for nb in result["undirected"][tx][:5]:
                    nodes_to_draw.add(nb)
        
        # Build subgraph
        G = nx.DiGraph()
        for a, b in zip(edges_df["txId1"], edges_df["txId2"]):
            if a in nodes_to_draw and b in nodes_to_draw:
                G.add_edge(a, b)
        
        if G.number_of_nodes() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            pos = nx.spring_layout(G, seed=42)
            flagged_set_local = set(result["flagged_df"]["txId"])
            node_colors = ["red" if n in flagged_set_local else "lightblue" for n in G.nodes()]
            nx.draw(
                G, pos, ax=ax,
                with_labels=True,
                node_color=node_colors,
                node_size=400,
                font_size=8,
                arrowsize=12,
                edge_color="gray"
            )
            ax.set_title("Sample Transaction Network (🔴 Red = Flagged Risky)")
            st.pyplot(fig)
        else:
            st.warning("Visualization ke liye koi edges nahi milin is sample mein.")
    
    except Exception as e:
        st.error(f"❌ Error aaya: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Kripya sidebar se dono CSV files upload karein taaki shuru kar saken.")
    
    st.subheader("📄 Sample Data Format")
    st.write("**Edgelist CSV (txId1, txId2):**")
    st.code("txId1,txId2\n230425980,5530458\n232022460,232438397\n")
    st.write("**Classes CSV (txId, class):**")
    st.code("txId,class\n230425980,unknown\n232438397,2\n")