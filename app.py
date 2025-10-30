# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Logistics Cost Scenario Lab", layout="wide")
st.title("📦 Logistics Cost Scenario Lab")
st.markdown("ML + Monte Carlo scenario tool adapted to your actual CSVs.")

# -------------------------
# Data loading (flexible paths)
# -------------------------
def find_base_dir():
    for base in ["/content", "/mnt/data", "."]:
        # check for at least one expected file
        if os.path.exists(os.path.join(base, "orders.csv")):
            return base
    return "/content"

BASE = find_base_dir()
st.info(f"Loading CSVs from `{BASE}`")

def safe_read(fname):
    p = os.path.join(BASE, fname)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.warning(f"Failed to read {fname}: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"File not found: {p}")
        return pd.DataFrame()

orders = safe_read("orders.csv")
inventory = safe_read("warehouse_inventory.csv")
routes = safe_read("routes_distance.csv")
costs = safe_read("cost_breakdown.csv")
delivery = safe_read("delivery_performance.csv")
fleet = safe_read("vehicle_fleet.csv")

# show a quick preview to confirm
st.subheader("Files detected (preview)")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("orders.csv")
    st.dataframe(orders.head(3))
with col2:
    st.write("warehouse_inventory.csv")
    st.dataframe(inventory.head(3))
with col3:
    st.write("routes_distance.csv")
    st.dataframe(routes.head(3))

col4, col5 = st.columns(2)
with col4:
    st.write("cost_breakdown.csv")
    st.dataframe(costs.head(3))
with col5:
    st.write("delivery_performance.csv")
    st.dataframe(delivery.head(3))
# -------------------------
# Prepare features (use actual column names)
# -------------------------
@st.cache_data
def prepare_features(orders, inventory, routes, costs, delivery):
    """
    Merge and aggregate the logistics datasets into a single table (city × product category),
    returning a numeric feature table with a total_cost column for ML modeling.
    """

    # ---------- Helper: lowercase + trim ----------
    def lc(df):
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    # ---------- Normalize all ----------
    orders_l = lc(orders)
    inv_l = lc(inventory)
    routes_l = lc(routes)
    costs_l = lc(costs)
    delivery_l = lc(delivery)

    # ================================================================
    # ORDERS
    # ================================================================
    if "origin" in orders_l.columns:
        orders_l = orders_l.rename(columns={"origin": "warehouse"})
    elif "destination" in orders_l.columns:
        orders_l = orders_l.rename(columns={"destination": "warehouse"})
    else:
        orders_l["warehouse"] = "UNKNOWN"

    if "product_category" in orders_l.columns:
        orders_l = orders_l.rename(columns={"product_category": "category"})
    else:
        pc = next((c for c in orders_l.columns if "category" in c), None)
        if pc:
            orders_l = orders_l.rename(columns={pc: "category"})
        else:
            orders_l["category"] = "MISC"

    # Ensure order_value numeric
    if "order_value" in orders_l.columns:
        orders_l["order_value"] = pd.to_numeric(orders_l["order_value"], errors="coerce").fillna(0)
    else:
        orders_l["order_value"] = 0.0

    if "order_id" not in orders_l.columns:
        orders_l["order_id"] = [f"O-UNK-{i}" for i in range(len(orders_l))]

    # ================================================================
    # INVENTORY
    # ================================================================
    inv = inv_l.copy()
    if "warehouse_location" in inv.columns:
        inv = inv.rename(columns={"warehouse_location": "warehouse"})
    elif "warehouse_id" in inv.columns:
        inv = inv.rename(columns={"warehouse_id": "warehouse"})

    if "product_category" in inv.columns:
        inv = inv.rename(columns={"product_category": "category"})
    elif "product" in inv.columns:
        inv = inv.rename(columns={"product": "category"})
    else:
        inv["category"] = "MISC"

    if "stock_level" in inv.columns:
        inv = inv.rename(columns={"stock_level": "qty"})
    elif "stock" in inv.columns:
        inv = inv.rename(columns={"stock": "qty"})
    else:
        inv["qty"] = np.nan

    if "storage_cost_pu_pm" in inv.columns:
        inv["avg_cost"] = pd.to_numeric(inv["storage_cost_pu_pm"], errors="coerce").fillna(0)
    else:
        inv["avg_cost"] = np.nan

    inv["qty"] = pd.to_numeric(inv["qty"], errors="coerce").fillna(0)

    inv_agg = inv.groupby(["warehouse", "category"], as_index=False).agg(
        qty=("qty", "sum"),
        avg_cost=("avg_cost", "mean")
    )

    # ================================================================
    # COSTS (order-level)
    # ================================================================
    costs_df = costs_l.copy()
    if "order_id" in costs_df.columns:
        for c in costs_df.columns:
            if c != "order_id":
                costs_df[c] = pd.to_numeric(costs_df[c], errors="coerce").fillna(0)
        costs_df["order_cost"] = costs_df.drop(columns=["order_id"]).select_dtypes(include=[np.number]).sum(axis=1)
        orders_cost = orders_l.merge(costs_df[["order_id", "order_cost"]], on="order_id", how="left")
    else:
        orders_cost = orders_l.copy()
        orders_cost["order_cost"] = 0.0

    # ================================================================
    # ROUTES & DELIVERY
    # ================================================================
    if "order_id" in routes_l.columns:
        if "distance_km" in routes_l.columns:
            routes_l["distance_km"] = pd.to_numeric(routes_l["distance_km"], errors="coerce").fillna(0)
        else:
            dcol = next((c for c in routes_l.columns if "dist" in c), None)
            if dcol:
                routes_l = routes_l.rename(columns={dcol: "distance_km"})
                routes_l["distance_km"] = pd.to_numeric(routes_l["distance_km"], errors="coerce").fillna(0)
        routes_join = orders_cost.merge(routes_l[["order_id", "distance_km"]], on="order_id", how="left")
    else:
        routes_join = orders_cost.copy()
        routes_join["distance_km"] = 0.0

    if "order_id" in delivery_l.columns:
        delivery_l["delivery_cost"] = pd.to_numeric(delivery_l.get("delivery_cost", 0), errors="coerce").fillna(0)
        routes_join = routes_join.merge(delivery_l[["order_id", "delivery_cost"]], on="order_id", how="left")
    else:
        routes_join["delivery_cost"] = 0.0

    # ================================================================
    # Combine all to order-level total_cost
    # ================================================================
    routes_join["order_total_cost"] = (
        routes_join["order_cost"].fillna(0)
        + routes_join["delivery_cost"].fillna(0)
        + routes_join["distance_km"].fillna(0) * 0.05  # simple transport factor
    )

    # ================================================================
    # Aggregate to warehouse (CITY) × category
    # ================================================================
    orders_agg = routes_join.groupby(["warehouse", "category"], as_index=False).agg(
        order_count=("order_id", "count"),
        demand_value=("order_value", "sum"),
        sum_order_cost=("order_total_cost", "sum"),
        avg_distance=("distance_km", "mean")
    )

    # ================================================================
    # Merge with inventory summary
    # ================================================================
    merged = inv_agg.merge(orders_agg, on=["warehouse", "category"], how="outer")

    merged["qty"] = merged["qty"].fillna(0)
    merged["avg_cost"] = merged["avg_cost"].fillna(0)
    merged["order_count"] = merged["order_count"].fillna(0)
    merged["demand_value"] = merged["demand_value"].fillna(0)
    merged["sum_order_cost"] = merged["sum_order_cost"].fillna(0)
    merged["avg_distance"] = merged["avg_distance"].fillna(0)

    # compute total_cost
    merged["total_cost"] = merged["sum_order_cost"]
    mask = merged["total_cost"] == 0
    merged.loc[mask, "total_cost"] = (
        merged.loc[mask, "qty"] * merged.loc[mask, "avg_cost"]
        + merged.loc[mask, "demand_value"] * merged.loc[mask, "avg_distance"] * 0.0001
    )

    merged = merged.fillna(0)
    merged["warehouse"] = merged["warehouse"].astype(str)
    merged["category"] = merged["category"].astype(str)

    return merged

df = prepare_features(orders, inventory, routes, costs, delivery)
st.subheader("Aggregated warehouse × category (prepared)")
st.dataframe(df.head(20))

# -------------------------
# Model training
# -------------------------
@st.cache_resource
def train_model(df):
    y = df["total_cost"].values
    X = df.drop(columns=["total_cost", "warehouse", "category"])  # keep numeric + categorical features
    # keep warehouse/category as categorical features
    X["warehouse"] = df["warehouse"]
    X["category"] = df["category"]

    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer([("num", num_pipe, num_feats),
                             ("cat", cat_pipe, cat_feats)])

    model = Pipeline([("pre", pre),
                      ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model, X, y, X_test, y_test

model, X_full, y_full, X_test, y_test = train_model(df)
st.success("RandomForest trained on aggregated table.")


# -------------------------
# Scenario UI
# -------------------------
st.sidebar.header("Scenario builder")
warehouses = sorted(df["warehouse"].unique().tolist())
warehouse_sel = st.sidebar.selectbox("Warehouse", warehouses)
categories = sorted(df[df["warehouse"] == warehouse_sel]["category"].unique().tolist())
category_sel = st.sidebar.selectbox("Category", ["--ALL--"] + categories)
action = st.sidebar.selectbox("Action", ["None", "Remove category (qty=0)", "Scale storage cost", "Scale demand"])
storage_mult = st.sidebar.number_input("Storage cost multiplier (1.0 = no change)", 0.0, 5.0, 1.0, 0.1)
demand_mult = st.sidebar.number_input("Demand multiplier (1.0 = no change)", 0.0, 5.0, 1.0, 0.1)
mc_iter = st.sidebar.slider("Monte Carlo iterations", 200, 3000, 1000, 200)
run_button = st.sidebar.button("Run scenario")

def baseline_kpis_from_df(df_in):
    return {
        "Total Operational Cost": df_in["total_cost"].sum(),
        "Total Storage Cost": (df_in["qty"] * df_in["avg_cost"] * 0.2).sum(),
        "Inventory Carrying Cost": (df_in["qty"] * df_in["avg_cost"] * 0.25).sum(),
        "Delivery Cost": df_in.get("sum_order_cost", 0).sum(),
        "Average Lead Time": 0,
        "Order Fill Rate": 0.95,
        "Stockouts": 0,
        "Inventory Turnover": df_in.get("demand_value", 0).sum() / (df_in["qty"].replace(0, np.nan).mean() + 1e-6),
        "Vehicle Utilization": 0.6,
        "CO2 Proxy": (df_in.get("demand_value", 0) * df_in.get("avg_distance", 0)).sum() * 0.0001
    }

baseline_kpis = baseline_kpis_from_df(df)
st.sidebar.markdown("### Baseline KPIs")
for k, v in baseline_kpis.items():
    st.sidebar.write(f"{k}: {v:,.2f}")

# -------------------------
# Scenario apply & Monte Carlo
# -------------------------
def apply_scenario(df_base, wh, cat, action, storage_mult, demand_mult):
    dfm = df_base.copy()
    if cat != "--ALL--":
        mask = (dfm["warehouse"] == wh) & (dfm["category"] == cat)
    else:
        mask = (dfm["warehouse"] == wh)

    if action == "Remove category (qty=0)":
        dfm.loc[mask, ["qty", "demand_value", "total_cost"]] = 0
    elif action == "Scale storage cost":
        dfm.loc[mask, "avg_cost"] = dfm.loc[mask, "avg_cost"] * storage_mult
        dfm.loc[mask, "total_cost"] = dfm.loc[mask, "qty"] * dfm.loc[mask, "avg_cost"] + dfm.loc[mask, "demand_value"] * dfm.loc[mask, "avg_distance"] * 0.0001
    elif action == "Scale demand":
        dfm.loc[mask, "demand_value"] = dfm.loc[mask, "demand_value"] * demand_mult
        dfm.loc[mask, "total_cost"] = dfm.loc[mask, "qty"] * dfm.loc[mask, "avg_cost"] + dfm.loc[mask, "demand_value"] * dfm.loc[mask, "avg_distance"] * 0.0001
    return dfm

def monte_carlo(df_mod, model, iterations=1000):
    rng = np.random.default_rng(42)
    kpi_rows = []

    # separate features
    Xbase = df_mod.drop(columns=["total_cost"]).copy()

    for _ in range(iterations):
        Xs = Xbase.copy()

        # random perturbations (simulate uncertainty)
        if "demand_value" in Xs.columns:
            Xs["demand_value"] *= rng.normal(1.0, 0.15, len(Xs))
        if "avg_cost" in Xs.columns:
            Xs["avg_cost"] *= rng.normal(1.0, 0.1, len(Xs))
        if "avg_distance" in Xs.columns:
            Xs["avg_distance"] *= rng.normal(1.0, 0.05, len(Xs))
        if "qty" in Xs.columns:
            Xs["qty"] *= rng.normal(1.0, 0.05, len(Xs))

        # clip numeric columns only
        num_cols = Xs.select_dtypes(include=[np.number]).columns
        Xs[num_cols] = Xs[num_cols].clip(lower=0)

        # ensure categorical columns still exist
        Xs["warehouse"] = df_mod["warehouse"]
        Xs["category"] = df_mod["category"]

        # predict new total_costs
        preds = model.predict(Xs)
        Xs["total_cost"] = preds

        # compute KPIs from these predictions
        kpi = baseline_kpis_from_df(Xs)
        kpi_rows.append(kpi)

    return pd.DataFrame(kpi_rows)


# -------------------------
# Run scenario
# -------------------------
if run_button:
    st.info("Running scenario and Monte Carlo simulation...")

    # --- Run scenario & simulation ---
    df_scenario = apply_scenario(df, warehouse_sel, category_sel, action, storage_mult, demand_mult)
    mc_df = monte_carlo(df_scenario, model, iterations=mc_iter)

    # --- Compute KPI changes ---
    scenario_mean = mc_df.mean()
    baseline_series = pd.Series(baseline_kpis)
    pct_change = (scenario_mean - baseline_series) / (baseline_series.replace(0, np.nan)) * 100

    results_df = pd.DataFrame({
        "Baseline": baseline_series,
        "Scenario Mean": scenario_mean,
        "Change (%)": pct_change
    })

    st.subheader("📊 KPI Comparison (Baseline → Scenario)")
    st.dataframe(results_df.style.format("{:,.2f}"))

    # --- KPI percentage change bar chart ---
    st.subheader("📈 KPI Percentage Change Overview")
    fig, ax = plt.subplots(figsize=(8, 5))
    pct_change.sort_values().plot(kind="barh", ax=ax, color=["red" if x > 0 else "green" for x in pct_change])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Percentage Change vs Baseline (%)")
    ax.set_ylabel("KPI")
    st.pyplot(fig)

    # --- Monte Carlo distributions ---
    st.subheader("Monte Carlo KPI Distributions")
    kpi_choice = st.selectbox("Select KPI to view distribution", results_df.index.tolist())
    fig2, ax2 = plt.subplots()
    ax2.hist(mc_df[kpi_choice].dropna(), bins=40, color="skyblue", edgecolor="gray")
    ax2.set_title(f"{kpi_choice} distribution (scenario)")
    st.pyplot(fig2)

st.markdown("---")
st.caption("App adapted to actual CSV columns you provided.")
