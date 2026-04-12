# NatWest ForeSight Enterprise: Semantic Data Mesh

[![Hackathon Submission](https://img.shields.io/badge/Hackathon-2024-blue.svg)](https://github.com/Sumedh08/ForeSight-Enterprise)
[![Security Audited](https://img.shields.io/badge/Security-Zero_Secrets-green.svg)](https://github.com/Sumedh08/ForeSight-Enterprise)

## Overview
NatWest ForeSight Enterprise is a high-performance "Intelligence Layer" designed for modern banking ecosystems. It transforms fragmented, heterogeneous data into actionable insights using a **Semantic Data Mesh** architecture. By bridging raw databases (PostgreSQL, MySQL, SQLite) with an AI-driven orchestration layer, it enables executives to query, forecast, and simulate business scenarios in plain English with enterprise-grade safety and precision.

The platform solves the "Data Silo" problem by providing a unified interface for natural language intelligence, automated machine learning (AutoML), and predictive scheduling.

---

## Features

### 🚀 Intelligence Core
- **Intent-Aware Routing**: Automatically identifies if a user wants a live SQL query, a future forecast, a scenario simulation, or anomaly detection.
- **Safe Text-to-SQL**: Integrated with **Wren AI** and **SQLGlot** for AST-level validation, ensuring only read-only, single-statement SELECTs reach your production DB.
- **Automated Forecasting**: Powered by **MindsDB**, the system auto-discovers temporal columns and trains predictors without manual feature engineering.
- **Scenario Simulation**: "What-if" analysis tool to model the impact of variable changes on key business metrics.

### 🛠️ Enterprise Infrastructure
- **Dynamic Discovery**: Apache Airflow DAGs automatically scan new data uploads to trigger semantic profiling and predictor training.
- **Connection Wizard**: Professional UI for managing multi-source database connections with encrypted credential handling.
- **Real-time Health Monitoring**: Integrated dashboard for tracking the status of LLM nodes, DB connections, and orchestration services.

---

## Tech Stack
- **Languages**: Python 3.11+, JavaScript
- **Frameworks**: FastAPI (Backend), Streamlit (Command Center)
- **AI/LLM**: NVIDIA NIM (Llama 3.1 / Qwen 2.5), SQLGlot (AST Security)
- **Data Layers**: Wren AI (Semantic Layer), MindsDB (AutoML), PostgreSQL (Primary)
- **Orchestration**: Apache Airflow, Redis, Celery
- **Containerization**: Docker & Docker Compose

---

## Install and Run Instructions

### Prerequisites
- Docker Desktop (Min 8GB RAM, 15GB Disk Space)
- NVIDIA API Key (for LLM inference)

### 1. One-Click Setup (Local Windows)
Run the bootstrap script to automate venv creation, dependency installation, and stack orchestration:
```bash
./bootstrap.bat
```

### 2. Manual Docker Startup
1. Copy `.env.example` to `.env` and set your `NVIDIA_API_KEY`.
2. Launch the enterprise stack:
```bash
docker compose up -d --build
```
3. Access the **Command Center** at `http://localhost:8501`.
4. Access the **API Documentation** at `http://localhost:8000/docs`.

---


## Future Roadmap
- **AI Agnosticism**: Expanding provider support to include **Ollama** for local-first inference and custom API keys for enterprise LLM gateways.
- **Mobile-First Optimization**: Developing a lightweight, "pocket analytics" PWA optimized for low-latency mobile access.
- **Cloud Native**: One-click deployment templates for AWS Fargate and Azure Container Apps.
- **Federated Queries**: Enabling cross-database joins via the Cube.js semantic bridge.

---

## Development & Security
- **DCO Compliant**: All contributions are signed-off.
- **Audit History**: Clean git history with zero hardcoded credentials.

**Contact**: Sumedh Ramesh Naidu ([sumedhramesh.naidu@2022vitstudent.ac.in](mailto:sumedhramesh.naidu@2022vitstudent.ac.in))
