# 🚚 Logistics Cost Scenario Lab

## 📘 Project Description
**Logistics Cost Scenario Lab** is an interactive **Streamlit** application designed for **NexGen Logistics** to perform predictive scenario analysis of its warehousing and delivery operations.

Using real operational data, it integrates **orders, inventory, routing, and cost breakdowns** to model total logistics cost at the level of **warehouse × product category**.  
The app enables managers to experiment with **“what-if” scenarios** — such as scaling demand or storage costs, or removing inventory categories — and immediately see the impact on key performance indicators (KPIs).

Under the hood:
- A **Random Forest model** predicts total logistics cost.
- **Monte Carlo simulation** quantifies uncertainty in the outcomes.

This tool transforms NexGen’s data into a **Cost Intelligence Platform**, directly supporting the company’s goal of **data-driven decision-making and 15–20% cost reduction**.

---

## 💡 Problem Statement
**NexGen Logistics**, a mid-sized logistics firm with **5 warehouses across India** and over **200 monthly orders** in diverse product categories, faces significant **cost pressures** and **operational inefficiencies**.

Managers need to understand how changes in **inventory levels** or **demand** at specific warehouses affect overall **costs** and **service levels**.

The unique problem addressed by this app is **predictive cost scenario analysis for multi-warehouse logistics**.

It answers:
> “If we adjust inventory or demand for a given product category in a warehouse, how will total operational cost and other KPIs (like storage cost, delivery cost, CO₂ proxy, etc.) change?”

By turning **reactive data** into **proactive insights**, the app helps NexGen move from ad-hoc decisions to a **predictive, data-driven culture**.

---

## ⚙️ Features and Functionality

### 🔗 Multi-Source Data Integration
- Loads and previews all provided CSV datasets (`orders`, `warehouse_inventory`, `routes_distance`, `cost_breakdown`, `delivery_performance`, etc.).
- Ensures the tool works with up-to-date logistics data.

### 📊 Data Aggregation
- Merges data by **warehouse (city)** and **product category**.
- Computes aggregate metrics: order counts, total demand value, and total route cost per warehouse–category pair.

### 🤖 Predictive Cost Modeling
- Trains a **Random Forest regression model** on the aggregated data.
- Predicts **total logistics cost** (including order, fuel, and delivery costs) for each warehouse–category segment.

### 🧩 Interactive Scenario Builder
In the sidebar, users can:
- **Remove category:** Set inventory and demand to zero (simulate discontinuation).  
- **Scale storage cost:** Apply a multiplier to the unit storage cost (simulate change in warehousing strategy).  
- **Scale demand:** Apply a multiplier to demand value (simulate surge or drop in sales).  

These inputs allow *“what-if”* exploration of inventory/demand changes.

### 🎲 Monte Carlo Simulation
- Runs multiple configurable trials with random perturbations (e.g. demand ±15%, storage cost ±10%, distances ±5%).  
- Quantifies **uncertainty** and **variability** in cost outcomes.

### 📈 KPI Dashboard
Displays:
- **Total Operational Cost**
- **Total Storage Cost**
- **Inventory Carrying Cost**
- **Delivery Cost**
- **Inventory Turnover**
- **CO₂ Proxy**, and more.

A **tabular comparison** shows baseline vs scenario values and **percentage change** for each KPI.

### 📉 Visualizations
- Interactive **bar charts** of KPI percentage changes (red/green bars).
- **Histograms** of Monte Carlo distributions for any selected KPI.
- Clearly highlights the **direction** and **uncertainty** of impacts under each scenario.

### 🧠 User Interface
- Built with **Streamlit** for a smooth, interactive experience.
- Dynamic updates in response to user selections.
- Minimal clicks: select → run scenario → view results.

---

## 🖥️ How to Run Locally

### 1️⃣ Environment Setup
Ensure **Python 3.7+** is installed.

### 2️⃣ Install Dependencies
Install requirements.txt

### 3️⃣ Add Data Files

Place all provided CSV files in the same directory as app.py, or inside a data/ folder (and update paths if needed):
Note that this app is only configured for the provided csv files as this is supposed to be a demo and not a production ready application.

- orders.csv
- warehouse_inventory.csv
- routes_distance.csv
- cost_breakdown.csv
- delivery_performance.csv

### 4️⃣ Launch App

Run the Streamlit app locally:

streamlit run app.py

The app will open automatically in your default web browser.

## 🖥️ App Screenshots
<img width="1440" height="900" alt="image" src="https://github.com/user-attachments/assets/4b1b3fc8-eab4-442e-8f5f-5db098f8b7aa" />
<img width="1440" height="900" alt="Bildschirmfoto 2025-10-30 um 6 37 17 PM" src="https://github.com/user-attachments/assets/4d048806-e07f-4560-ae20-3721f222af1d" />
<img width="1440" height="900" alt="Bildschirmfoto 2025-10-30 um 6 37 25 PM" src="https://github.com/user-attachments/assets/01a85e05-876f-4312-84c4-4a328baf42fe" />
<img width="1440" height="900" alt="Bildschirmfoto 2025-10-30 um 6 37 32 PM" src="https://github.com/user-attachments/assets/b79f5983-7f1b-4675-a976-3814b542a882" />
<img width="1440" height="900" alt="Bildschirmfoto 2025-10-30 um 6 37 42 PM" src="https://github.com/user-attachments/assets/5650182c-cba8-4a57-9288-1d8475f2e622" />




