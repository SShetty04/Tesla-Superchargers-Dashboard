# =============================================================================
# Tesla Supercharger — Procurement Intelligence Dashboard
# Run: py -m streamlit run app.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Procurement Intelligence Dashboard for Tesla Superchargers",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
.kpi-card {
    background: #1a1d27;
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid #2d3748;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #f3f4f6;
    line-height: 1.2;
    margin-top: 4px;
}
.kpi-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #9ca3af;
    margin-top: 8px;
    border-top: 1px solid #2d3748;
    padding-top: 8px;
    flex-wrap: wrap;
    gap: 6px;
}
.kpi-row b { color: #f3f4f6; }
.insight {
    background: #131624;
    border-left: 3px solid #E31937;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
    color: #d1d5db;
    margin: 6px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLUSTER_ORDER = [
    "Tier 1 - Priority Fleet",
    "Tier 2 - Growth Markets",
    "Tier 3 - Upgrade Pipeline",
    "Tier 4 - Stable Network",
    "Tier 5 - Retirement Pipeline",
]
CLUSTER_COLORS = {
    "Tier 1 - Priority Fleet":      "#E31937",
    "Tier 2 - Growth Markets":      "#F59E0B",
    "Tier 3 - Upgrade Pipeline":    "#8B5CF6",
    "Tier 4 - Stable Network":      "#6366F1",
    "Tier 5 - Retirement Pipeline": "#6B7280",
}
# Tiers where V3 upgrade is the primary procurement lever
UPGRADE_TIERS = ["Tier 3 - Upgrade Pipeline", "Tier 5 - Retirement Pipeline"]
# States in the High EV Adoption Corridor (used to derive corridor from state selection)
HIGH_EV_STATES = {"CA", "CO", "FL", "NJ", "TX", "WA"}
CAT_COLS = [
    "Energy_spend", "Preventive_maint_spend", "Corrective_maint_spend",
    "Site_services_spend", "Network_spend",
]
CAT_LABELS = ["Energy", "Preventive Maint", "Corrective Maint", "Site Services", "Network"]
CAT_COLORS = ["#E31937", "#F59E0B", "#10B981", "#6366F1", "#8B5CF6"]

FEATURES  = ["version", "Corridor", "Size_band", "Age_band",
             "Stalls", "Capacity_kw_total", "site_age_years"]
CAT_FEATS = ["version", "Corridor", "Size_band", "Age_band"]

DARK = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
BG   = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


# ── Formatters ────────────────────────────────────────────────────────────────
def fmt_m(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v * 100:.1f}%"


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("clustered_superchargers_AMER_v2.xlsx", engine="openpyxl")
    # All financial and numeric columns are already native float/int in the Excel file.

    # Network-wide percentile ranks for risk scoring (not within-tier)
    for metric, asc in [
        ("Profit_margin",    False),   # lower margin  → higher risk rank
        ("TCO_per_kwh_usd", True),    # higher TCO    → higher risk rank
        ("site_age_years",  True),    # older site    → higher risk rank
        ("Payback_years",   True),    # longer payback → higher risk rank
    ]:
        df[f"_r_{metric}"] = df[metric].rank(pct=True, ascending=asc)

    df["Risk_Score"] = (
        df["_r_Profit_margin"]    * 0.35 +
        df["_r_TCO_per_kwh_usd"] * 0.35 +
        df["_r_site_age_years"]  * 0.20 +
        df["_r_Payback_years"]   * 0.10
    )
    df["Risk_Tier"] = pd.cut(
        df["Risk_Score"],
        bins=[0, 0.40, 0.65, 1.01],
        labels=["Low", "Medium", "High"],
    ).astype(str).replace("nan", pd.NA)

    return df


# ── ML Models ─────────────────────────────────────────────────────────────────
def _fit_encoders(df):
    """Fit LabelEncoders on full dataset so all categories are known."""
    enc = {}
    for col in CAT_FEATS:
        le = LabelEncoder()
        le.fit(df[col].dropna().astype(str).unique())
        enc[col] = le
    return enc


def _encode_with(X: pd.DataFrame, enc: dict) -> pd.DataFrame:
    X = X.copy()
    for col in CAT_FEATS:
        le = enc[col]
        vals = X[col].astype(str)
        # Map unseen values to first known class
        safe = vals.apply(lambda v: v if v in le.classes_ else le.classes_[0])
        X[col] = le.transform(safe)
    return X


@st.cache_resource
def train_opex_model(_df):
    enc  = _fit_encoders(_df)
    data = _df[FEATURES + ["Annual_opex_usd"]].dropna()
    X    = _encode_with(data[FEATURES], enc)
    y    = data["Annual_opex_usd"]
    mdl  = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=3)
    mdl.fit(X, y)
    preds = mdl.predict(X)
    return mdl, enc, data.index.tolist(), preds, r2_score(y, preds)


@st.cache_resource
def train_rev_model(_df):
    enc  = _fit_encoders(_df)
    data = _df[FEATURES + ["Annual_revenue_usd"]].dropna()
    X    = _encode_with(data[FEATURES], enc)
    y    = data["Annual_revenue_usd"]
    mdl  = GradientBoostingRegressor(n_estimators=300, random_state=42,
                                      learning_rate=0.05, max_depth=4)
    mdl.fit(X, y)
    return mdl, enc


def predict_one(model, enc, version, corridor, size_band, age_band, stalls, cap_kw, age_yrs):
    row = pd.DataFrame([{
        "version": version, "Corridor": corridor,
        "Size_band": size_band, "Age_band": age_band,
        "Stalls": stalls, "Capacity_kw_total": cap_kw,
        "site_age_years": age_yrs,
    }])
    return float(model.predict(_encode_with(row, enc))[0])


# ── Bootstrap ─────────────────────────────────────────────────────────────────
df = load_data()
opex_model, opex_enc, opex_idx, opex_preds, opex_r2 = train_opex_model(df)
rev_model,  rev_enc  = train_rev_model(df)

# Attach predictions and gap columns to master df
df["Should_Cost"]  = np.nan
df.loc[opex_idx, "Should_Cost"] = opex_preds
df["Opex_Gap"]     = df["Annual_opex_usd"] - df["Should_Cost"]
df["Opex_Gap_Pct"] = df["Opex_Gap"] / df["Should_Cost"]

# State cost index: avg opex gap for each state (controls for site type via should-cost)
# Positive = sites in this state run above their should-cost on average
state_gap_map = df.groupby("State")["Opex_Gap_Pct"].mean().to_dict()

# Pre-compute V3 upgrade metrics for all Tier 3/5 sites (vectorized batch predict)
# Avoids 810 row-by-row predict_one() calls every time Tab B renders
_upgrade_mask = df["Cluster"].isin(UPGRADE_TIERS)
_v3_feats = df[_upgrade_mask][FEATURES].copy()
_v3_feats["version"]           = "V3"
_v3_feats["Capacity_kw_total"] = _v3_feats["Stalls"] * 250
df.loc[_upgrade_mask, "_v3_opex"] = opex_model.predict(_encode_with(_v3_feats, opex_enc))
df.loc[_upgrade_mask, "_v3_rev"]  = rev_model.predict(_encode_with(_v3_feats, rev_enc))
df["_curr_prof"]            = df["Annual_revenue_usd"] - df["Annual_opex_usd"]
df["_v3_prof"]              = df["_v3_rev"] - df["_v3_opex"]
df["_incr_profit"]          = df["_v3_prof"] - df["_curr_prof"]
df["Upgrade_Cost"]          = df["Stalls"] * 150_000
df["Upgrade_Payback_years"] = df["Upgrade_Cost"] / df["_incr_profit"].clip(lower=1)
df["Incr_Annual_Profit"]    = df["_incr_profit"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Procurement Intelligence Dashboard for Tesla Superchargers")
    st.divider()

    nav = st.radio(
        "Navigate",
        ["Descriptive", "Prescriptive", "Predictive"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Filters**")

    sel_cluster = st.multiselect(
        "Cluster",
        options=CLUSTER_ORDER,
        default=CLUSTER_ORDER,
    )
    sel_corridor = st.multiselect(
        "Corridor",
        options=sorted(df["Corridor"].dropna().unique().tolist()),
        default=sorted(df["Corridor"].dropna().unique().tolist()),
    )
    sel_version = st.multiselect(
        "Version",
        options=sorted(df["version"].dropna().unique().tolist()),
        default=sorted(df["version"].dropna().unique().tolist()),
    )
    sel_state = st.multiselect(
        "State",
        options=sorted(df["State"].dropna().unique().tolist()),
        placeholder="All states",
    )

    st.divider()

# Apply global filters
fdf = df[
    df["Cluster"].isin(sel_cluster) &
    df["Corridor"].isin(sel_corridor) &
    df["version"].isin(sel_version)
].copy()
if sel_state:
    fdf = fdf[fdf["State"].isin(sel_state)]

st.sidebar.caption(f"Sites in view: **{len(fdf):,}**")

if len(fdf) == 0:
    st.warning("No sites match current filters. Please widen your selection.")
    st.stop()


# =============================================================================
# SECTION 1 — DESCRIPTIVE
# =============================================================================
def render_descriptive(fdf):
    st.markdown("## Network Overview")

    # KPI Cards
    stats = (
        fdf.groupby("Cluster")
        .agg(
            N       =("Supercharger",     "count"),
            AvgRev  =("Annual_revenue_usd","mean"),
            AvgOpex =("Annual_opex_usd",   "mean"),
            AvgMargin=("Profit_margin",    "mean"),
            AvgTCO  =("TCO_per_kwh_usd",   "mean"),
        )
        .reindex([c for c in CLUSTER_ORDER if c in fdf["Cluster"].unique()])
    )

    if "selected_tier" not in st.session_state:
        st.session_state["selected_tier"] = None

    cols = st.columns(len(stats))
    for i, (cluster, row) in enumerate(stats.iterrows()):
        color = CLUSTER_COLORS[cluster]
        with cols[i]:
            st.markdown(
                f'<div class="kpi-card" style="border-top:3px solid {color}">'
                f'<div class="kpi-label">{cluster}</div>'
                f'<div class="kpi-value">{fmt_m(row.AvgOpex)}</div>'
                f'<div style="font-size:12px;color:#9ca3af;margin-top:2px">'
                f'avg annual opex &nbsp;·&nbsp; {row.N:,.0f} sites</div>'
                f'<div class="kpi-row">'
                f'<span>Avg Rev: <b>{fmt_m(row.AvgRev)}</b></span>'
                f'<span>Margin: <b>{pct(row.AvgMargin)}</b></span>'
                f'<span>TCO/kWh: <b>${row.AvgTCO:.3f}</b></span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            is_active = st.session_state["selected_tier"] == cluster
            label = "▲ Close" if is_active else "View sites →"
            if st.button(label, key=f"tier_btn_{i}"):
                st.session_state["selected_tier"] = None if is_active else cluster
                st.rerun()

    # Tier drill-down
    active_tier = st.session_state["selected_tier"]
    if active_tier and active_tier in fdf["Cluster"].values:
        color = CLUSTER_COLORS[active_tier]
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:6px 14px;'
            f'margin:12px 0 4px 0;font-size:15px;font-weight:600;">'
            f'{active_tier} - Site Detail</div>',
            unsafe_allow_html=True,
        )
        tier_sites = fdf[fdf["Cluster"] == active_tier][[
            "Supercharger", "State", "City", "version", "Stalls",
            "Capacity_kw_total", "site_age_years",
            "Annual_revenue_usd", "Annual_opex_usd", "Profit_margin",
            "Opex_per_stall", "TCO_per_kwh_usd", "Opex_Gap_Pct", "Risk_Tier",
        ]].copy().sort_values("Opex_Gap_Pct", ascending=False)

        tier_sites["Annual_revenue_usd"] = tier_sites["Annual_revenue_usd"].apply(fmt_m)
        tier_sites["Annual_opex_usd"]    = tier_sites["Annual_opex_usd"].apply(fmt_m)
        tier_sites["Opex_per_stall"]     = tier_sites["Opex_per_stall"].apply(fmt_m)
        tier_sites["Profit_margin"]      = tier_sites["Profit_margin"].apply(pct)
        tier_sites["TCO_per_kwh_usd"]    = tier_sites["TCO_per_kwh_usd"].apply(lambda v: f"${v:.3f}")
        tier_sites["Opex_Gap_Pct"]       = tier_sites["Opex_Gap_Pct"].apply(
            lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "–"
        )
        tier_sites["site_age_years"]     = tier_sites["site_age_years"].apply(lambda v: f"{v:.1f}")
        tier_sites.columns = [
            "Site", "State", "City", "Version", "Stalls", "Capacity (kW)",
            "Age (yrs)", "Revenue", "Opex", "Margin",
            "Opex/Stall", "TCO/kWh", "Cost Variance", "Risk Tier",
        ]
        st.dataframe(tier_sites, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns([3, 2])

    with col1:
        cat = fdf.groupby("Cluster")[CAT_COLS].sum().reset_index()
        fig = go.Figure()
        for label, col_name, color in zip(CAT_LABELS, CAT_COLS, CAT_COLORS):
            fig.add_trace(go.Bar(
                name=label,
                x=cat["Cluster"],
                y=cat[col_name] / 1e6,
                marker_color=color,
                hovertemplate=f"<b>{label}</b><br>$%{{y:.1f}}M<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack",
            title="Category Spend by Cluster ($M)",
            legend=dict(orientation="h", y=-0.28),
            height=390,
            xaxis=dict(tickangle=0),
            yaxis=dict(title="$M"),
            margin=dict(l=40, r=20, t=40, b=10),
            **DARK,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        opex_share = fdf.groupby("Cluster")["Annual_opex_usd"].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=opex_share["Cluster"],
            values=opex_share["Annual_opex_usd"],
            hole=0.55,
            marker_colors=[CLUSTER_COLORS[c] for c in opex_share["Cluster"]],
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig2.update_layout(
            title="Opex Concentration by Cluster",
            showlegend=False,
            height=390,
            margin=dict(l=20, r=20, t=40, b=10),
            **DARK,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Geographic map
    st.markdown("### Site Map")
    map_df = fdf.dropna(subset=["Latitude", "Longitude"])
    fig3 = px.scatter_geo(
        map_df,
        lat="Latitude",
        lon="Longitude",
        color="Cluster",
        size="Annual_revenue_usd",
        size_max=18,
        hover_name="Supercharger",
        hover_data={
            "Cluster": True,
            "Annual_revenue_usd": ":$,.0f",
            "Profit_margin": ":.1%",
            "version": True,
            "Latitude": False,
            "Longitude": False,
        },
        color_discrete_map=CLUSTER_COLORS,
        projection="albers usa",
        scope="usa",
        height=520,
    )
    fig3.update_layout(
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            lakecolor="#1e2030",
            landcolor="#2d3748",
            showland=True,
            showlakes=True,
            subunitcolor="#4b5563",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        **DARK,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Tier definitions
    st.markdown("### Tier Definitions")
    tier_defs = [
        ("Tier 1 - Priority Fleet",      "#E31937", "V3 chargers in high EV adoption corridors. Highest revenue, core network backbone."),
        ("Tier 2 - Growth Markets",       "#F59E0B", "V3 chargers in lower EV adoption corridors. Strong hardware, developing demand."),
        ("Tier 3 - Upgrade Pipeline",     "#8B5CF6", "V1/V2 chargers in high EV adoption corridors. High upgrade ROI given strong local demand."),
        ("Tier 4 - Stable Network",       "#6366F1", "V2 chargers in lower EV adoption corridors. Steady performers, lower strategic priority."),
        ("Tier 5 - Retirement Pipeline",  "#6B7280", "V1 chargers in lower EV adoption corridors. Oldest hardware, lowest demand - candidates for retirement or upgrade."),
    ]
    tier_cols = st.columns(5)
    for col, (name, color, desc) in zip(tier_cols, tier_defs):
        with col:
            st.markdown(
                f'<div style="border-top:3px solid {color};padding:10px 0 4px">'
                f'<div style="font-size:13px;font-weight:600;color:{color};margin-bottom:6px">{name}</div>'
                f'<div style="font-size:12px;color:#9ca3af;line-height:1.5">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# SECTION 2 — PRESCRIPTIVE
# =============================================================================
def render_prescriptive(df_full, fdf):
    st.markdown("## Procurement Intelligence (Should-Cost & Benchmarks)")

    # ── Total savings opportunity banner ──────────────────────────────────────
    pos_gap       = fdf[fdf["Opex_Gap"] > 0].dropna(subset=["Opex_Gap", "Opex_Gap_Pct"])
    total_savings = pos_gap["Opex_Gap"].sum()
    sites_above   = len(pos_gap)
    avg_gap_pct   = pos_gap["Opex_Gap_Pct"].mean() if len(pos_gap) > 0 else 0
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Addressable Savings", fmt_m(total_savings),
              delta=f"{sites_above} sites above benchmark", delta_color="inverse")
    s2.metric("Sites Above Should-Cost", f"{sites_above:,}",
              delta=f"{sites_above/len(fdf)*100:.0f}% of network in view")
    s3.metric("Avg Cost Variance", f"{avg_gap_pct*100:+.1f}%",
              delta="vs peer-based benchmark", delta_color="inverse")

    tab_calc, tab_bench = st.tabs(["Should-Cost Calculator", "Benchmark & Upgrade ROI"])

    # ── Tab A: Should-Cost Calculator ─────────────────────────────────────────
    with tab_calc:
        col_form, col_res = st.columns([1, 2])

        with col_form:
            st.markdown("**Configure Site Profile**")
            sc_ver    = st.selectbox("Charger Version", ["V1", "V2", "V3"])
            sc_state  = st.selectbox("State", sorted(df_full["State"].dropna().unique().tolist()))
            sc_cor    = "High EV Adoption Corridor" if sc_state in HIGH_EV_STATES else "Lower EV Adoption Corridor"
            st.caption(f"Corridor: **{sc_cor}**")
            sc_stalls   = st.slider("Stalls", 2, 80, 10)
            sc_size     = "Large" if sc_stalls >= 62 else ("Medium" if sc_stalls >= 31 else "Small")
            st.caption(f"Size Band: **{sc_size}**")
            sc_age      = st.slider("Site Age (yrs)", 1, 14, 7)
            sc_age_band = "Old" if sc_age > 8 else ("Mid" if sc_age >= 3 else "New")
            st.caption(f"Age Band: **{sc_age_band}** - moderate influence on should-cost (~3–5% of model weight)")
            ver_kw    = {"V1": 72, "V2": 150, "V3": 250}
            sc_cap    = sc_stalls * ver_kw.get(sc_ver, 150)

        with col_res:
            pred_opex  = predict_one(opex_model, opex_enc,
                                     sc_ver, sc_cor, sc_size, sc_age_band,
                                     sc_stalls, sc_cap, float(sc_age))
            pred_rev   = predict_one(rev_model, rev_enc,
                                     sc_ver, sc_cor, sc_size, sc_age_band,
                                     sc_stalls, sc_cap, float(sc_age))

            # Apply state cost index: avg opex gap for sites in this state
            state_adj  = state_gap_map.get(sc_state, 0.0)
            adj_opex   = pred_opex * (1 + state_adj)
            pred_marg  = (pred_rev - adj_opex) / pred_rev if pred_rev > 0 else 0

            # Map to expected tier based on version + corridor + size
            if sc_ver == "V3" and sc_cor == "High EV Adoption Corridor":
                comp_cluster = "Tier 1 - Priority Fleet"
            elif sc_ver == "V3":
                comp_cluster = "Tier 2 - Growth Markets"
            elif sc_ver in ["V1", "V2"] and sc_cor == "High EV Adoption Corridor":
                comp_cluster = "Tier 3 - Upgrade Pipeline"
            elif sc_ver == "V2":
                comp_cluster = "Tier 4 - Stable Network"
            else:
                comp_cluster = "Tier 5 - Retirement Pipeline"

            cl_peers_avg = df_full[
                (df_full["Cluster"] == comp_cluster) &
                (df_full["Size_band"] == sc_size) &
                (df_full["Corridor"] == sc_cor)
            ]["Annual_opex_usd"].mean()
            cl_avg  = cl_peers_avg if not np.isnan(cl_peers_avg) else df_full[df_full["Cluster"] == comp_cluster]["Annual_opex_usd"].mean()
            gap_pct = (adj_opex - cl_avg) / cl_avg if cl_avg > 0 else 0

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Should-Cost Opex", fmt_m(adj_opex),
                          delta=f"{gap_pct*100:+.1f}% vs peer avg",
                          delta_color="inverse")
            with r2:
                st.metric("Predicted Revenue", fmt_m(pred_rev))
            with r3:
                st.metric("Implied Margin", pct(pred_marg))

            # State cost index info
            if abs(state_adj) > 0.005:
                direction = "above" if state_adj > 0 else "below"
                st.info(
                    f"**Cluster:** {comp_cluster} &nbsp;·&nbsp; Peer avg ({sc_size} / {sc_cor.replace(' Adoption Corridor', '')}): {fmt_m(cl_avg)}  \n"
                    f"State index [{sc_state}]: {state_adj*100:+.1f}% - {sc_state} sites "
                    f"typically run {abs(state_adj)*100:.1f}% {direction} their model baseline. "
                    f"Base: {fmt_m(pred_opex)} → State-adjusted: {fmt_m(adj_opex)}"
                )
            else:
                st.info(f"**Cluster:** {comp_cluster} &nbsp;·&nbsp; Peer avg ({sc_size} / {sc_cor.replace(' Adoption Corridor', '')}): {fmt_m(cl_avg)}")

            # Category breakdown — ratios from peer sites, not hardcoded
            peer_pool = df_full[
                (df_full["version"] == sc_ver) &
                (df_full["Corridor"] == sc_cor) &
                (df_full["Size_band"] == sc_size)
            ]
            if len(peer_pool) >= 3:
                cat_ratios = [(peer_pool[c] / peer_pool["Annual_opex_usd"]).mean() for c in CAT_COLS]
            else:
                cat_ratios = [(df_full[c] / df_full["Annual_opex_usd"]).mean() for c in CAT_COLS]
            cat_names = [f"{l} ({r*100:.0f}%)" for l, r in zip(CAT_LABELS, cat_ratios)]
            cat_vals  = [adj_opex * r for r in cat_ratios]
            fig_cats = go.Figure(go.Bar(
                x=cat_vals, y=cat_names, orientation="h",
                marker_color=CAT_COLORS,
                text=[fmt_m(v) for v in cat_vals],
                textposition="outside",
            ))
            fig_cats.update_layout(
                title="Should-Cost Opex - Category Breakdown",
                xaxis=dict(title="Estimated Annual Spend"),
                height=230,
                margin=dict(l=0, r=80, t=40, b=0),
                **DARK,
            )
            st.plotly_chart(fig_cats, use_container_width=True)

            # Peer sites
            peers = df_full[
                (df_full["version"] == sc_ver) &
                (df_full["Corridor"] == sc_cor) &
                (df_full["Size_band"] == sc_size)
            ].nsmallest(5, "Opex_per_stall")[
                ["Supercharger", "State", "Annual_opex_usd", "Opex_per_stall",
                 "Annual_revenue_usd", "Profit_margin"]
            ].copy()

            if not peers.empty:
                st.markdown("**5 most efficient peer sites (same version / corridor / size) - sorted by Opex/Stall:**")
                peers["Annual_opex_usd"]    = peers["Annual_opex_usd"].apply(fmt_m)
                peers["Opex_per_stall"]     = peers["Opex_per_stall"].apply(fmt_m)
                peers["Annual_revenue_usd"] = peers["Annual_revenue_usd"].apply(fmt_m)
                peers["Profit_margin"]      = peers["Profit_margin"].apply(pct)
                peers.columns = ["Site", "State", "Opex", "Opex/Stall", "Revenue", "Margin"]
                st.dataframe(peers, use_container_width=True, hide_index=True)

        # ── Site-to-Site Comparison ────────────────────────────────────────────
        st.divider()
        st.markdown("### Site-to-Site Comparison")
        selected_sites = st.multiselect(
            "Select sites to compare (2–5 recommended)",
            options=sorted(df_full["Supercharger"].dropna().tolist()),
            max_selections=6,
            key="site_compare",
        )

        if len(selected_sites) < 2:
            st.caption("Select at least 2 sites to activate the comparison panel.")
        else:
            cmp_df = df_full[df_full["Supercharger"].isin(selected_sites)].set_index("Supercharger")
            # Use selected order
            cmp_df = cmp_df.loc[[s for s in selected_sites if s in cmp_df.index]]

            # ── Comparison table (metrics as rows, sites as columns) ───────────
            def _build_comparison_sections(rows_df):
                return {
                    "Profile": [
                        ("State",          lambda r: r["State"]),
                        ("Cluster",        lambda r: r["Cluster"]),
                        ("Version",        lambda r: r["version"]),
                        ("Stalls",         lambda r: str(int(r["Stalls"]))),
                        ("Capacity (kW)",  lambda r: f"{int(r['Capacity_kw_total']):,}"),
                        ("Age (yrs)",      lambda r: f"{r['site_age_years']:.1f}"),
                        ("Corridor",       lambda r: r["Corridor"].replace(" Adoption Corridor", "")),
                    ],
                    "Financials": [
                        ("Annual Revenue", lambda r: fmt_m(r["Annual_revenue_usd"])),
                        ("Annual Opex",    lambda r: fmt_m(r["Annual_opex_usd"])),
                        ("Profit Margin",  lambda r: pct(r["Profit_margin"])),
                        ("Opex / Stall",   lambda r: fmt_m(r["Opex_per_stall"])),
                        ("Rev / Stall",    lambda r: fmt_m(r["Revenue_per_stall"])),
                        ("TCO / kWh",      lambda r: f"${r['TCO_per_kwh_usd']:.3f}"),
                        ("Payback (yrs)",  lambda r: f"{r['Payback_years']:.1f}" if r["Payback_years"] < 99 else ">99"),
                    ],
                    "Should-Cost Analysis": [
                        ("Should-Cost",    lambda r: fmt_m(r["Should_Cost"]) if pd.notna(r.get("Should_Cost")) else "–"),
                        ("Opex Gap ($)",   lambda r: fmt_m(r["Opex_Gap"])    if pd.notna(r.get("Opex_Gap"))    else "–"),
                        ("Opex Gap (%)",   lambda r: f"{r['Opex_Gap_Pct']*100:+.1f}%" if pd.notna(r.get("Opex_Gap_Pct")) else "–"),
                        ("Risk Tier",      lambda r: r.get("Risk_Tier", "–") if pd.notna(r.get("Risk_Tier")) else "–"),
                    ],
                    "Category Spend": [
                        ("Energy",          lambda r: fmt_m(r["Energy_spend"])),
                        ("Preventive Maint",lambda r: fmt_m(r["Preventive_maint_spend"])),
                        ("Corrective Maint",lambda r: fmt_m(r["Corrective_maint_spend"])),
                        ("Site Services",   lambda r: fmt_m(r["Site_services_spend"])),
                        ("Network",         lambda r: fmt_m(r["Network_spend"])),
                    ],
                }

            for sec_name, metrics in _build_comparison_sections(cmp_df).items():
                st.markdown(f"**{sec_name}**")
                records = {label: {s: fn(cmp_df.loc[s]) for s in cmp_df.index}
                           for label, fn in metrics}
                st.dataframe(pd.DataFrame(records).T, use_container_width=True)

            # ── Grouped bar: key financials ────────────────────────────────────
            bar_metrics = ["Annual_revenue_usd", "Annual_opex_usd", "Should_Cost"]
            bar_labels  = ["Annual Revenue", "Annual Opex", "Should-Cost"]
            bar_colors  = ["#10B981", "#E31937", "#F59E0B"]

            fig_cmp = go.Figure()
            for col, label, color in zip(bar_metrics, bar_labels, bar_colors):
                fig_cmp.add_trace(go.Bar(
                    name=label,
                    x=selected_sites,
                    y=cmp_df[col].values / 1e6,
                    marker_color=color,
                    text=[f"${v/1e6:.1f}M" for v in cmp_df[col].values],
                    textposition="outside",
                ))
            fig_cmp.update_layout(
                barmode="group",
                title="Revenue vs Opex vs Should-Cost ($M)",
                yaxis=dict(title="$M"),
                height=340,
                legend=dict(orientation="h", y=-0.20),
                margin=dict(l=10, r=10, t=40, b=10),
                **DARK,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # ── Stacked bar: category spend as % of opex ──────────────────────
            fig_cat = go.Figure()
            for label, col, color in zip(CAT_LABELS, CAT_COLS, CAT_COLORS):
                pct_vals = (cmp_df[col] / cmp_df["Annual_opex_usd"] * 100).round(1)
                fig_cat.add_trace(go.Bar(
                    name=label,
                    x=selected_sites,
                    y=pct_vals.values,
                    marker_color=color,
                    text=[f"{v:.0f}%" for v in pct_vals.values],
                    textposition="inside",
                ))
            fig_cat.update_layout(
                barmode="stack",
                title="Opex Category Mix (% of total opex)",
                yaxis=dict(title="% of Opex", ticksuffix="%"),
                height=300,
                legend=dict(orientation="h", y=-0.28),
                margin=dict(l=10, r=10, t=40, b=10),
                **DARK,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            # ── Cost variance callout ─────────────────────────────────────────
            gap_series = cmp_df["Opex_Gap_Pct"].dropna()
            if not gap_series.empty:
                worst_site    = gap_series.idxmax()
                best_site     = gap_series.idxmin()
                cost_variance = (gap_series.max() - gap_series.min()) * 100
                worst_gap     = gap_series.max() * 100
                worst_gap_usd = cmp_df.loc[worst_site, "Opex_Gap"]
                if worst_gap > 5:
                    st.warning(
                        f"**Priority for cost recovery:** {worst_site} is running {worst_gap:+.1f}% ({fmt_m(worst_gap_usd)}) above its should-cost benchmark.  \n"
                        f"**Lowest-cost operator:** {best_site} at {gap_series.min()*100:+.1f}% vs benchmark - use as internal best-practice reference.  \n"
                        f"**Cost variance across selection:** {cost_variance:.1f}pp - closing this gap on the highest-cost site represents the immediate savings opportunity."
                    )
                else:
                    st.info(
                        f"Selected sites show low cost variance ({cost_variance:.1f}pp). "
                        f"Lowest-cost operator: **{best_site}** - no material savings opportunity within this peer group."
                    )

    # ── Tab B: Benchmark & ROI ─────────────────────────────────────────────────
    with tab_bench:
        legacy = fdf[fdf["Cluster"].isin(UPGRADE_TIERS)].dropna(subset=["Upgrade_Payback_years"])
        legacy = legacy.nsmallest(30, "Upgrade_Payback_years")

        if not legacy.empty:
            legacy["Payback_Tier"] = pd.cut(
                legacy["Upgrade_Payback_years"],
                bins=[0, 5, 8, float("inf")],
                labels=["< 5 yrs (Act Now)", "5–8 yrs (Plan)", "> 8 yrs (Deprioritize)"],
            ).astype(str)

            median_profit = legacy["Incr_Annual_Profit"].median()
            median_cost   = legacy["Upgrade_Cost"].median()

            fig2 = px.scatter(
                legacy,
                x="Incr_Annual_Profit",
                y="Upgrade_Cost",
                color="Payback_Tier",
                size="Stalls",
                size_max=22,
                hover_name="Supercharger",
                hover_data={
                    "State": True,
                    "version": True,
                    "Stalls": True,
                    "Upgrade_Cost": ":$,.0f",
                    "Incr_Annual_Profit": ":$,.0f",
                    "Upgrade_Payback_years": ":.1f",
                    "Payback_Tier": False,
                },
                color_discrete_map={
                    "< 5 yrs (Act Now)":      "#10B981",
                    "5–8 yrs (Plan)":         "#F59E0B",
                    "> 8 yrs (Deprioritize)": "#6B7280",
                },
                category_orders={"Payback_Tier": ["< 5 yrs (Act Now)", "5–8 yrs (Plan)", "> 8 yrs (Deprioritize)"]},
                title="V3 Upgrade ROI Quadrant - Top 30 Sites",
                labels={
                    "Incr_Annual_Profit": "Incremental Annual Profit ($)",
                    "Upgrade_Cost":       "Upgrade Cost ($)",
                },
                height=520,
                template="plotly_dark",
            )
            fig2.add_vline(x=median_profit, line_dash="dash", line_color="#4b5563",
                           annotation_text="Median profit gain", annotation_position="top")
            fig2.add_hline(y=median_cost, line_dash="dash", line_color="#4b5563",
                           annotation_text="Median upgrade cost", annotation_position="right")
            fig2.update_layout(
                xaxis=dict(tickprefix="$", tickformat=",.0f"),
                yaxis=dict(tickprefix="$", tickformat=",.0f"),
                legend=dict(orientation="h", y=-0.12, title="Payback Tier"),
                **BG,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No Upgrade Pipeline / Retirement Pipeline sites in current filter.")

        # Outlier table
        st.markdown("### Top Procurement Opportunities - Highest Opex vs. Should-Cost Gap")
        outliers = fdf.dropna(subset=["Opex_Gap_Pct"]).nlargest(15, "Opex_Gap_Pct")[
            ["Supercharger", "State", "Cluster", "version",
             "Annual_opex_usd", "Should_Cost", "Opex_Gap", "Opex_Gap_Pct"]
        ].copy()
        outliers["Annual_opex_usd"] = outliers["Annual_opex_usd"].apply(fmt_m)
        outliers["Should_Cost"]     = outliers["Should_Cost"].apply(fmt_m)
        outliers["Opex_Gap"]        = outliers["Opex_Gap"].apply(fmt_m)
        outliers["Opex_Gap_Pct"]    = outliers["Opex_Gap_Pct"].apply(
            lambda v: f"{v*100:+.1f}%" if not pd.isna(v) else "–"
        )
        outliers.columns = ["Site", "State", "Cluster", "Version",
                            "Actual Opex", "Should-Cost", "Gap ($)", "Gap (%)"]
        st.dataframe(outliers, use_container_width=True, hide_index=True)

    # ── Corridor Reference ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Corridor Definitions")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**High EV Adoption Corridor** - 6 states  \n"
            "CA · CO · FL · NJ · TX · WA  \n"
            "<span style='font-size:12px;color:#9ca3af'>High charger utilisation, stronger revenue per stall, "
            "priority for V3 investment and fleet contract negotiation.</span>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "**Lower EV Adoption Corridor** - 46 states  \n"
            "AK · AL · AR · AZ · CT · DC · DE · GA · HI · IA · ID · IL · IN · KS · KY · "
            "LA · MA · MD · ME · MI · MN · MO · MS · MT · NC · ND · NE · NH · NM · NV · "
            "NY · OH · OK · OR · PA · PR · RI · SC · SD · TN · UT · VA · VT · WI · WV · WY  \n"
            "<span style='font-size:12px;color:#9ca3af'>Lower utilisation density. Higher scrutiny on "
            "expansion capex; Tier 4 and Tier 5 sites predominantly in this corridor.</span>",
            unsafe_allow_html=True,
        )


# =============================================================================
# SECTION 3 — PREDICTIVE
# =============================================================================
def render_predictive(df_full, fdf):
    st.markdown("## Predictive Analytics - Risk & Cost Driver Analysis")

    tab_risk, tab_traj = st.tabs(["Risk & Upgrade Simulator", "Cost Driver Analysis"])

    # ── Tab A: Cost Driver Analysis ────────────────────────────────────────────
    with tab_traj:
        st.markdown("### Cost Driver Analysis")
        st.markdown(
            '<div class="insight">Identifies which operational factors are systematically '
            'associated with sites running above their should-cost benchmark, then lets you '
            'drill into any individual site to pinpoint its specific cost drivers.</div>',
            unsafe_allow_html=True,
        )

        gap_df = fdf.dropna(subset=["Opex_Gap_Pct"])

        c1, c2 = st.columns(2)

        with c1:
            ver_gap = (
                gap_df.groupby("version")["Opex_Gap_Pct"]
                .mean().reset_index().sort_values("Opex_Gap_Pct")
            )
            fig1 = go.Figure(go.Bar(
                x=ver_gap["Opex_Gap_Pct"], y=ver_gap["version"], orientation="h",
                marker_color=[
                    "#E31937" if v > 0.02 else "#F59E0B" if v > 0 else "#10B981"
                    for v in ver_gap["Opex_Gap_Pct"]
                ],
                text=[f"{v*100:+.1f}%" for v in ver_gap["Opex_Gap_Pct"]],
                textposition="outside",
            ))
            fig1.add_vline(x=0, line_color="#6b7280", line_width=1)
            fig1.update_layout(
                title="Avg Opex Gap by Charger Version",
                xaxis=dict(tickformat="+.0%", title="Avg gap vs should-cost"),
                yaxis_title="", height=220,
                margin=dict(l=10, r=70, t=40, b=10), **DARK,
            )
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            tier_gap = (
                gap_df.groupby("Cluster")["Opex_Gap_Pct"]
                .mean()
                .reindex([c for c in CLUSTER_ORDER if c in gap_df["Cluster"].unique()])
                .reset_index()
            )
            fig2 = go.Figure(go.Bar(
                x=tier_gap["Opex_Gap_Pct"], y=tier_gap["Cluster"], orientation="h",
                marker_color=[CLUSTER_COLORS.get(c, "#6b7280") for c in tier_gap["Cluster"]],
                text=[f"{v*100:+.1f}%" for v in tier_gap["Opex_Gap_Pct"]],
                textposition="outside",
            ))
            fig2.add_vline(x=0, line_color="#6b7280", line_width=1)
            fig2.update_layout(
                title="Avg Opex Gap by Tier",
                xaxis=dict(tickformat="+.0%", title="Avg gap vs should-cost"),
                yaxis_title="", height=220,
                margin=dict(l=10, r=70, t=40, b=10), **DARK,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # State-level gap — top 20 most overspending states (min 3 sites)
        state_gap_df = (
            gap_df.groupby("State")["Opex_Gap_Pct"]
            .agg(mean="mean", count="count")
            .reset_index()
        )
        state_gap_df = (
            state_gap_df[state_gap_df["count"] >= 3]
            .sort_values("mean", ascending=False)
            .head(20)
        )
        if not state_gap_df.empty:
            fig3 = go.Figure(go.Bar(
                x=state_gap_df["mean"],
                y=state_gap_df["State"],
                orientation="h",
                marker_color=[
                    "#E31937" if v > 0.02 else "#F59E0B" if v > 0 else "#10B981"
                    for v in state_gap_df["mean"]
                ],
                text=[f"{v*100:+.1f}%" for v in state_gap_df["mean"]],
                textposition="outside",
                customdata=state_gap_df["count"],
                hovertemplate="<b>%{y}</b><br>Avg gap: %{x:.1%}<br>Sites: %{customdata}<extra></extra>",
            ))
            fig3.add_vline(x=0, line_color="#6b7280", line_width=1)
            fig3.update_layout(
                title="Top 20 States by Avg Opex Overspend vs Should-Cost  (min 3 sites)",
                xaxis=dict(tickformat="+.0%", title="Avg gap vs should-cost"),
                yaxis_title="", height=420,
                margin=dict(l=10, r=80, t=40, b=10), **DARK,
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── Site-level drilldown ───────────────────────────────────────────────
        st.markdown("### Site-Level Cost Driver Drilldown")
        drilldown_opts = (
            fdf.dropna(subset=["Opex_Gap_Pct"])
            .sort_values("Opex_Gap_Pct", ascending=False)["Supercharger"]
            .tolist()
        )
        if not drilldown_opts:
            st.info("No sites with valid cost gap data in current filter.")
        else:
            sel_site = st.selectbox(
                "Select site to investigate",
                drilldown_opts, key="drilldown_site",
            )
            srow     = df_full[df_full["Supercharger"] == sel_site].iloc[0]
            cl_peers = df_full[df_full["Cluster"] == srow["Cluster"]]

            d1, d2 = st.columns([1, 2])

            with d1:
                st.markdown(
                    f"**{sel_site}**  \n"
                    f"{srow['State']} · {srow['version']} · {int(srow['Stalls'])} stalls  \n"
                    f"{srow['Cluster']}"
                )
                gap_val = srow["Opex_Gap_Pct"]
                gap_usd = srow["Opex_Gap"]
                st.metric(
                    "Opex Gap vs Should-Cost",
                    f"{gap_val*100:+.1f}%",
                    delta=fmt_m(gap_usd),
                    delta_color="inverse",
                )
                s_idx = state_gap_map.get(srow["State"], 0.0)
                if abs(s_idx) > 0.01:
                    st.caption(
                        f"State cost index [{srow['State']}]: {s_idx*100:+.1f}% - "
                        "typical location premium/discount for this state."
                    )

                # Factor comparison table
                site_corr_pct = srow["Corrective_maint_spend"] / srow["Annual_opex_usd"] * 100
                peer_corr_pct = (cl_peers["Corrective_maint_spend"] / cl_peers["Annual_opex_usd"] * 100).median()
                factor_rows = []
                for label, site_v, med_v, fmt_fn in [
                    ("Opex / Stall",    srow["Opex_per_stall"],  cl_peers["Opex_per_stall"].median(),  fmt_m),
                    ("TCO / kWh",       srow["TCO_per_kwh_usd"], cl_peers["TCO_per_kwh_usd"].median(), lambda v: f"${v:.3f}"),
                    ("Profit Margin",   srow["Profit_margin"],   cl_peers["Profit_margin"].median(),   pct),
                    ("Site Age (yrs)",  srow["site_age_years"],  cl_peers["site_age_years"].median(),  lambda v: f"{v:.1f}"),
                    ("Corrective %",    site_corr_pct,           peer_corr_pct,                        lambda v: f"{v:.1f}%"),
                ]:
                    dev_str = f"{(site_v - med_v) / abs(med_v) * 100:+.1f}%" if med_v else "—"
                    factor_rows.append({
                        "Factor":         label,
                        "This Site":      fmt_fn(site_v),
                        "Cluster Median": fmt_fn(med_v),
                        "vs Median":      dev_str,
                    })
                st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)

            with d2:
                # % deviation from cluster median for each metric
                labels, deviations, colors = [], [], []
                deviation_data = [
                    ("Opex / Stall",    srow["Opex_per_stall"],  cl_peers["Opex_per_stall"].median(),  True),
                    ("TCO / kWh",       srow["TCO_per_kwh_usd"], cl_peers["TCO_per_kwh_usd"].median(), True),
                    ("Profit Margin",   srow["Profit_margin"],   cl_peers["Profit_margin"].median(),   False),
                    ("Site Age",        srow["site_age_years"],  cl_peers["site_age_years"].median(),  True),
                    ("Corrective %",    site_corr_pct,           peer_corr_pct,                        True),
                    ("Revenue / Stall", srow["Revenue_per_stall"], cl_peers["Revenue_per_stall"].median(), False),
                ]
                for label, site_v, med_v, higher_is_worse in deviation_data:
                    if med_v and med_v != 0:
                        dev    = (site_v - med_v) / abs(med_v) * 100
                        is_bad = (dev > 0) == higher_is_worse
                        labels.append(label)
                        deviations.append(dev)
                        colors.append(
                            "#E31937" if is_bad and abs(dev) > 5 else
                            "#F59E0B" if abs(dev) > 2 else "#10B981"
                        )

                fig_dev = go.Figure(go.Bar(
                    x=deviations, y=labels, orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.1f}%" for v in deviations],
                    textposition="outside",
                ))
                fig_dev.add_vline(x=0, line_color="#6b7280", line_width=1)
                fig_dev.update_layout(
                    title=f"{sel_site} - % Deviation from Cluster Median",
                    xaxis=dict(title="% deviation from cluster median", ticksuffix="%"),
                    yaxis_title="", height=340,
                    margin=dict(l=10, r=80, t=40, b=10), **DARK,
                )
                st.plotly_chart(fig_dev, use_container_width=True)

                bad_factors = [labels[i] for i in range(len(labels)) if colors[i] == "#E31937"]
                if bad_factors:
                    st.markdown(
                        f'<div class="insight">Primary cost drivers vs cluster median: '
                        f'<b>{"  ·  ".join(bad_factors)}</b>. These are the areas where '
                        f'procurement investigation is most warranted for this site.</div>',
                        unsafe_allow_html=True,
                    )

    # ── Tab B: Risk & Upgrade Simulator ───────────────────────────────────────
    with tab_risk:
        st.markdown("### V3 Upgrade Simulator")
        st.caption("Predict post-upgrade metrics for any V1 / V2 Legacy / Large site.")
        sim_col, cost_col = st.columns([2, 1])
        _upgrade_cost = None
        _new_payback  = None

        with sim_col:
            legacy_sites = df_full[
                df_full["Cluster"].isin(UPGRADE_TIERS)
            ]["Supercharger"].sort_values().tolist()

            if not legacy_sites:
                st.info("No V1/V2 Legacy sites in dataset.")
            else:
                site = st.selectbox("Select Site", legacy_sites, key="sim_site")
                row  = df_full[df_full["Supercharger"] == site].iloc[0]

                curr_opex = row["Annual_opex_usd"]
                curr_rev  = row["Annual_revenue_usd"]
                curr_prof = curr_rev - curr_opex
                curr_marg = row["Profit_margin"]
                stalls    = int(row["Stalls"])

                new_opex = predict_one(
                    opex_model, opex_enc, "V3", row["Corridor"],
                    row["Size_band"], row["Age_band"], stalls,
                    stalls * 250, row["site_age_years"]
                )
                new_rev = predict_one(
                    rev_model, rev_enc, "V3", row["Corridor"],
                    row["Size_band"], row["Age_band"], stalls,
                    stalls * 250, row["site_age_years"]
                )
                new_prof = new_rev - new_opex
                new_marg = (new_rev - new_opex) / new_rev if new_rev > 0 else 0

                upgrade_cost  = stalls * 150_000
                incr_profit   = new_prof - curr_prof
                new_payback   = upgrade_cost / max(incr_profit, 1)
                _upgrade_cost = upgrade_cost
                _new_payback  = new_payback

                if new_payback < 5:
                    st.success("Strong candidate - payback under 5 years")
                elif new_payback < 8:
                    st.warning("Moderate case - payback 5–8 years")
                else:
                    st.error("Weak case - payback exceeds 8 years")

                st.markdown(
                    f"**{site}** - {row['State']} &nbsp;·&nbsp; "
                    f"{row['version']} &nbsp;·&nbsp; {stalls} stalls"
                )

                for label, before, after, is_pct in [
                    ("Annual Revenue", curr_rev,  new_rev,  False),
                    ("Annual Opex",    curr_opex, new_opex, False),
                    ("Profit Margin",  curr_marg, new_marg, True),
                ]:
                    b_fmt = pct(before) if is_pct else fmt_m(before)
                    a_fmt = pct(after)  if is_pct else fmt_m(after)
                    if is_pct:
                        delta = f"{(after - before)*100:+.1f}pp"
                    else:
                        delta = f"{(after - before)/before*100:+.1f}%" if before else "–"
                    ca, cb, cc = st.columns([2, 2, 1])
                    ca.markdown(f"{label}")
                    cb.markdown(f"{b_fmt} → {a_fmt}")
                    cc.markdown(f"`{delta}`")

        with cost_col:
            if _upgrade_cost is not None:
                st.markdown("**Investment Summary**")
                st.metric("Upgrade Cost (est.)", fmt_m(_upgrade_cost))
                st.caption("Based on $150K / stall industry estimate. Actual cost varies by site, contractor, and state.")
                st.metric(
                    "Incremental Payback",
                    f"{_new_payback:.1f} yrs" if _new_payback > 0 and _new_payback < 100 else "N/A"
                )

        st.divider()
        st.markdown("### Site Risk Map")
        risk_df = fdf.dropna(subset=["Risk_Score", "TCO_per_kwh_usd", "Profit_margin"])
        risk_df = risk_df[risk_df["Risk_Tier"].notna()].copy()

        if not risk_df.empty:
            fig = px.scatter(
                risk_df,
                x="TCO_per_kwh_usd",
                y="Profit_margin",
                color="Risk_Tier",
                size="Risk_Score",
                size_max=16,
                color_discrete_map={
                    "Low":    "#10B981",
                    "Medium": "#F59E0B",
                    "High":   "#E31937",
                },
                category_orders={"Risk_Tier": ["Low", "Medium", "High"]},
                hover_name="Supercharger",
                hover_data={"State": True, "Cluster": True,
                            "Risk_Score": ":.2f", "version": True,
                            "Risk_Tier": False},
                opacity=0.75,
                render_mode="svg",
                title="Site Risk Map: TCO/kWh vs Profit Margin",
                labels={
                    "TCO_per_kwh_usd": "TCO per kWh ($)",
                    "Profit_margin":   "Profit Margin",
                },
                height=400,
                template="plotly_dark",
            )
            fig.update_layout(yaxis=dict(tickformat=".0%"), **BG)
            st.plotly_chart(fig, use_container_width=True)

            n_high = (risk_df["Risk_Tier"] == "High").sum()
            n_med  = (risk_df["Risk_Tier"] == "Medium").sum()
            n_low  = (risk_df["Risk_Tier"] == "Low").sum()
            r1, r2, r3 = st.columns(3)
            r1.metric("High Risk Sites",   f"{n_high:,}")
            r2.metric("Medium Risk Sites", f"{n_med:,}")
            r3.metric("Low Risk Sites",    f"{n_low:,}")
        else:
            st.info("Insufficient risk data for current filter.")


# =============================================================================
# ROUTING
# =============================================================================
if nav == "Descriptive":
    render_descriptive(fdf)
elif nav == "Prescriptive":
    render_prescriptive(df, fdf)
elif nav == "Predictive":
    render_predictive(df, fdf)
