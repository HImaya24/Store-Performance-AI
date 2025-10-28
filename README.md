# Store-Performance-AI

# 🛍️ Store Performance AI  
[![Build Status](https://img.shields.io/github/actions/workflow/status/HImaya24/Store-Performance-AI/ci.yml?branch=main)](https://github.com/HImaya24/Store-Performance-AI/actions)  
[![License](https://img.shields.io/github/license/HImaya24/Store-Performance-AI.svg)](LICENSE)  
[![Issues](https://img.shields.io/github/issues/HImaya24/Store-Performance-AI.svg)](https://github.com/HImaya24/Store-Performance-AI/issues)  
[![Stars](https://img.shields.io/github/stars/HImaya24/Store-Performance-AI.svg)](https://github.com/HImaya24/Store-Performance-AI/stargazers)

## 🎯 Overview
**Store Performance AI** is a distributed microservices platform that processes real-time retail transaction data while preserving customer privacy.  
It empowers retailers with **AI-driven insights**, **semantic search**, and **real-time KPIs** through a **scalable, privacy-aware architecture**.

---

## 📘 Table of Contents
1. [Background](#-background)  
2. [Key Features](#-key-features)  
3. [System Architecture](#-system-architecture)  
4. [Quick Start](#-quick-start)  
5. [Docker Deployment](#-docker-deployment)  
6. [Dashboard](#-dashboard)  
7. [Configuration](#-configuration)  
8. [Privacy & Security](#-privacy--security)  
9. [Use Cases](#-use-cases)  
10. [Contributing](#-contributing)  
11. [License](#-license)  
12. [Support](#-support)

---

## 🧩 Background
### The Challenge  
Traditional analytics systems struggle to generate **real-time retail insights** while maintaining **privacy and compliance**.

### The Solution  
A **modular microservices-based system** where each agent specializes in a specific task:
| Agent | Port | Responsibility |
|-------|------|----------------|
| **Collector** | 8100 | Data ingestion & PII protection |
| **Coordinator** | 8110 | Workflow orchestration |
| **Analyzer** | 8101 | Business insights generation |
| **KPI Engine** | 8102 | Performance metrics tracking |
| **IR Search** | 8103 | Semantic pattern discovery |

---

## ⭐ Key Features
- 🧠 **AI-Powered Chatbot** – Conversational business intelligence  
- 🔍 **Semantic IR Search** – Understands natural language queries  
- 📈 **Real-Time KPI Dashboard** – Live performance metrics  
- ⚡ **Automated Insights** – Trend and anomaly detection  
- 🔒 **PII Protection** – Automatic redaction for sensitive data  
- 🌐 **Scalable Architecture** – Microservices-based for flexibility  
- 🛒 **Product Discovery** – Intelligent transaction exploration  

---

## 🏗️ System Architecture

### Data Flow
Data Sources → Collector → Coordinator → Analyzer → Insights
↓ ↓
KPI Engine ←→ IR Search → Dashboard


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- React

### Installation
1. **Clone Repository**
bash
git clone <repository-url>
cd retail-analytics-platform

2. **Backend Setup**
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

3. **Frontend Setup**
cd frontend
npm install
npm start

4. **Start Microservices**

Each service runs in a separate terminal:

python collector/main.py
python coordinator/main.py
python analyzer/main.py
python kpi/main.py
python ir_search/main.py

## 🐳 Docker Deployment
Build and Run (All Services)
docker-compose up --build

Stop Containers
docker-compose down


The docker-compose.yml should define all core agents and frontend services for seamless orchestration.

## 📊 Dashboard

Access the interactive dashboard at:
👉 http://localhost:3000

## 🔧 Configuration

Create a .env file in the root directory with the following:

COLLECTOR_URL=http://localhost:8100
COORDINATOR_URL=http://localhost:8110
ANALYZER_URL=http://localhost:8101
KPI_URL=http://localhost:8102
API_KEY=demo-key

## 🛡️ Privacy & Security

🧹 Automatic PII redaction

🔑 Token-based authentication

🧩 CORS protection

🧱 Input validation to prevent injection or malformed data

## 📈 Use Cases

🏬 Store Performance Analysis

👥 Customer Behavior Insights

📆 Seasonal Trend Detection

💳 Payment Method Analytics

📦 Inventory Optimization

## 🤝 Contributing

1.Fork the repo

2.Create a feature branch:

git checkout -b feature/your-feature


3.Commit your changes:

git commit -m "Add new feature"


4.Push to your branch:

git push origin feature/your-feature


5.Open a Pull Request

## 📄 License

This project is licensed under the MIT License.
See the LICENSE
 file for details.

## 🆘 Support

💬 Open an Issue

📘 Check the /docs folder

🔍 Review API specifications

Built with ❤️ using FastAPI
, React
, and Material-UI
