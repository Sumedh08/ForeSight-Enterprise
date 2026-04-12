# NatWest ForeSight Enterprise AI 🚀

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Enterprise Ready](https://img.shields.io/badge/Architecture-Data%20Mesh-green)](https://github.com/Sumedh08/NatWest-ForeSight-Enterprise)

## i. Overview
NatWest ForeSight is an enterprise-grade predictive analytics platform designed to decentralize data intelligence. It solves the "Data Silo" problem by providing a schema-agnostic semantic layer that automatically discovers, models, and forecasts time-series data across any database table. Intended for financial analysts and data engineers, it enables decision-making through deterministic Text-to-SQL and automated In-Database Machine Learning.

---

## ii. Features
- **Deterministic Text-to-SQL**: Powered by **Wren AI**, ensuring natural language queries are grounded in your specific business logic.
- **Automated In-Database ML**: Integrates **MindsDB** to auto-identify time-series candidates and train predictive models without manual intervention.
- **Dynamic Semantic Layer**: Utilizes **Cube.js** to dynamically discover table schemas and generate metrics (AVG, SUM, COUNT) at runtime.
- **Enterprise Workflow**: Orchestrated by **Apache Airflow** for continuous data mesh synchronization and predictor training.
- **Schema-Agnostic Design**: Zero hardcoding. Works out-of-the-box with any uploaded CSV, Excel, or existing database table.
- **SQL Guard & PII Masking**: Built-in safety layers to prevent SQL injection and mask sensitive data in analytical responses.

---

## iii. Install and Run Instructions

### Prerequisites
- **Windows OS**
- **Docker Desktop** (Running with at least 8GB RAM allocated)
- **Python 3.11+**
- **NVIDIA NIM API Key** (for LLM reasoning)

### Step-by-Step Setup
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Sumedh08/NatWest-ForeSight-Enterprise
    cd NatWest-ForeSight-Enterprise
    ```
2.  **Run the One-Click Bootstrap**:
    Double-click or run `bootstrap.bat` in your terminal. This will:
    - Create a Python virtual environment (`.venv`).
    - Install all necessary dependencies.
    - Start the 10+ Docker containers in the enterprise stack.
    - Automate the initial data migration to PostgreSQL.
3.  **Configure API Keys**:
    Open the newly created `.env` file and paste your `NVIDIA_API_KEY`.
4.  **Start the Assistant**:
    The bootstrap script will automatically open the **FastAPI Backend** and the **Streamlit Frontend**.

---

## iv. Tech Stack
- **Languages**: Python, JavaScript, SQL.
- **Backend Framework**: FastAPI (Asynchronous analytical orchestration).
- **Orchestration**: Apache Airflow 2.9 (Dynamic DAGs for data mesh).
- **ML/AI Engine**: MindsDB (Automated Predictors), NVIDIA NIM (Llama 3.1 reasoning).
- **Semantic Layer**: Cube.js (Dynamic schema discovery), Wren AI (Deterministic NL2SQL).
- **Databases**: PostgreSQL 15 (Enterprise Storage), Redis (Caching), Qdrant (Vector Store).
- **Frontend**: Streamlit + Altair (Interactive predictive visualizations).

---

## v. Usage Examples

### 1. Natural Language Data Discovery
> **Question**: "What was the average mortgage rate in 2024?"
> **Output**: Generated SQL, summary statistics, and an interactive trend chart.

### 2. Autonomous Time-Series Forecasting
> **Question**: "Predict the mortage rate for the next 12 months."
> **Output**: A deep-inference forecast with 80% confidence intervals, powered by MindsDB.

### 3. Schema-Agnostic Upload
Simply upload a CSV in the sidebar. The system will:
1. Load it into Postgres.
2. Auto-profile it in Cube.js.
3. Schedule a MindsDB training job via Airflow.

---

## vi. Architecture Notes
The architecture follows a **Semantic Data Mesh** pattern:
1.  **Storage Layer**: Decentralized source data is consolidated into a persistent PostgreSQL instance.
2.  **Semantic Layer**: Cube.js acts as the universal translator, mapping raw tables into business dimensions and metrics dynamically.
3.  **Inference Layer**: Wren AI performs deterministic Text-to-SQL translation by "grounding" the LLM on the Cube.js metadata, preventing hallucinations common in generic LLM-SQL bridges.
4.  **Predictive Layer**: MindsDB encapsulates the complexity of ML, allowing the system to treat "Future Predictions" as simple SQL `JOIN` operations.

---

## vii. Limitations & Future Work
- **Hardware Footprint**: Requires significant local resources (8GB+ RAM). Future versions will support remote compute offloading.
- **Real-time Streaming**: Currently optimized for batch and interval-based synchronization (Airflow).
- **Future Improvements**: We plan to implement Multi-Source Federated Queries across different DB types (Snowflake, BigQuery) using the existing Cube.js bridge.

---

## 📜 Compliance & DCO
By submitting this project, I confirm compliance with the Apache License 2.0 and the Developer Certificate of Origin (DCO).
**Signed-off-by**: Sumedh Ramesh Naidu <sumedhramesh.naidu@2022vitstudent.ac.in>
