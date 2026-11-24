#  AI-powered Intrusion Detection Systems — HIDS

AI-powered Intrusion Detection Systems is an advanced, hybrid **Host-based Intrusion Detection System (HIDS)** designed using Python, integrating signature-based **File Integrity Monitoring (FIM)** with anomaly-based **Machine Learning (ML)** detection.

Inspired by industry-standard tools like **OSSEC** and **Tripwire**, HIDS adds modern machine learning techniques To detect any suspicious activity and anomalous system-call patterns. It demonstrates sophisticated techniques for real-time file system monitoring, cryptographic baseline creation, and behavioural analysis.

The system leverages the **XGBoost** algorithm, **n-gram** feature extraction, and the **Watchdog** library to detect unauthorised modifications and malicious system call sequences with minimal performance impact.

>  **Disclaimer:** This project is intended solely for educational and research purposes. Do not use it for any malicious activities. The author is NOT responsible for any misuse of this software.

---

##  Features

### 1. Hybrid Detection Framework
- Integrates signature-based FIM with anomaly-based ML detection for comprehensive threat coverage.
- Modular architecture where FIM and ML engines operate cohesively through a consolidated alert and log management system.

### 2. Real-Time File Monitoring
- Uses the `watchdog` library to monitor file system events (creation, modification, deletion, movement) in real-time.
- Context-aware severity assessment (e.g., modifications in `/etc` or `System32` trigger High/Critical alerts).
- Immediate desktop pop-up notifications for Critical and High severity events.

### 3. Machine Learning Engine
- **XGBoost** classifier trained on system call sequences to detect anomalous behaviour.
- `CountVectorizer` for **2–4-gram ** feature extraction from system call logs.
- **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance.
- **GridSearchCV** for hyperparameter optimisation to maximise accuracy and reduce false positives.

### 4. Comprehensive Alert Management
- Multi-channel alerting: unique IDs, logging to `security_alerts.log`, real-time GUI updates.
- Color-coded severity levels: 🔴 Critical | 🟠 High | 🟡 Medium | 🔵 Low
- JSON export of alert history for forensic analysis and auditing.

### 5. Intuitive Graphical User Interface
- Tabbed **Tkinter** interface: Dashboard, File Monitoring, AI Detection, Security Alerts, System Logs, Settings.
- One-click operations: Start/Stop Monitoring, Create Baseline, Run Integrity Checks.

---

##  Development Environment

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.x |
| GUI Framework | Tkinter |
| Machine Learning | XGBoost, Scikit-learn, Imbalanced-learn (SMOTE) |
| File Monitoring | Watchdog |
| Data Processing & Visualisation | Pandas, NumPy, Matplotlib, Seaborn |
| Utilities | Joblib (model serialization), Hashlib (SHA-256), JSON, Logging |
| Testing Environment | VirtualBox / VMware (isolated testing) |
| OS| Ubuntu, Windows

---
## Core Components
hids_app.py
Initialises the Tkinter GUI and orchestrates all subsystems. Loads hids_config.json to determine monitored paths (e.g., /etc, /bin on Linux or C:\Windows\System32 on Windows) and excluded paths (e.g., /tmp, /proc). Manages the lifecycle of the FileIntegrityMonitor, MLIntrusionDetector, and AlertManager classes.

## FileIntegrityMonitor

- Baseline Creation: Recursively hashes all files in monitored directories using SHA-256 in 4KB chunks. Records file size, modification time, and permissions → saved to baseline.json.
- Integrity Checking: Compares current file states against the baseline. Detects hash mismatch, size change, permission change, and file deletion — categorised by severity.
- Real-Time Monitoring: Spawns a watchdog observer to trigger immediate alerts for suspicious activity in protected directories.
---
## MLIntrusionDetector
- Model Training: End-to-end ML pipeline — loading system call sequences, n-gram feature extraction, SMOTE balancing, and GridSearchCV tuning. Evaluated on the ADFA-LD dataset with 97.42% classification accuracy.
- Prediction: Evaluates incoming sequences (e.g., ptrace, mmap, execve, chroot). Severity based on probability thresholds (>0.8 = Critical, >0.6 = High).
- Demo Mode: Generates realistic demo data and trains a lightweight model on the fly — test the system immediately without external training data.

## AlertManager
- Aggregates alerts, assigns sequential IDs, and updates the GUI.
- Logs everything to security_alerts.log with timestamps, severity, paths, and descriptions.
- Triggers desktop pop-up notifications for Critical/High alerts.
---
##  Execution Theory
## On Launch
The GUI initialises and loads hids_config.json. The Dashboard shows the status of File Monitoring, Baseline, ML Model, and active alerts.

## "Create Baseline"
1. Iterates through configured monitored_paths (skipping excluded_paths).
2. Calculates SHA-256 hash + metadata (size, modified time, permissions) for each file.
3. Serialises data to baseline.json.
4. GUI updates to "BASELINE CREATED".
---
## Real-Time Monitoring
1. "Start File Monitoring" spawns the watchdog observer.
2. On file create/modify/delete/move, the handler evaluates the path.
3. Critical keywords (system32, etc., bin) elevate severity to High/Critical.
4. Alert is logged, colour-coded in the GUI, and a desktop notification fires.
---
## ML Detection
1. "Train ML Model" runs the full pipeline (or auto-creates a demo model if no dataset is provided).
2. "Start ML Monitoring" periodically analyses system call sequences.
3. Suspicious patterns generate ML Intrusion alerts with severity classification.
---
##  Installation & Setup

### Prerequisites
- **Python 3.7** or higher
- **Tkinter** (Usually included with standard Python installations. If missing on Linux, install via `sudo apt-get install python3-tk` or `sudo dnf install python3-tkinter`).

### 1. Install Dependencies
The application requires several libraries for machine learning, file monitoring, and data visualization. You can install them manually using pip:

```bash
pip install watchdog psutil scikit-learn xgboost imbalanced-learn joblib seaborn
```
> **Note:** You can also skip this step and run the script directly; the application will detect missing dependencies and prompt you to install them automatically.
---
### 2. Run the Application
Navigate to the project directory and run the main script:
```bash
python "HIDS 19.1.1.py"
```
## Usage Guide
Upon launching, you will be greeted by a tabbed Graphical User Interface (GUI). The system is designed to be cross-platform compatible (Windows and Ubuntu) and will automatically adjust default monitored system paths based on your OS.

## 1. Dashboard
The central hub for monitoring the system's security posture.
- System Status: View the real-time state of File Monitoring, Baseline creation, ML Model loading, and System Integrity.
- Quick Actions: One-click buttons to Start/Stop Monitoring, Create Baselines, Run Integrity Checks, and Train ML Models.
  
## 2. File Monitoring (FIM)
This module uses the watchdog library and cryptographic hashing to detect unauthorized file changes.

- Add/Remove Path: Specify which directories to protect (e.g., /etc, /bin, C:\Windows\System32).
- Create Baseline: (Required First Step) Generates a secure JSON snapshot (baseline.json) of the hashes and metadata of all files in your monitored paths.
- Start/Stop Monitoring: Activates real-time, event-driven monitoring. Any file creation, modification, or deletion in protected directories will trigger an immediate alert.
- Run Integrity Check: Performs a manual, full-scan comparison of current files against the saved baseline to detect silent tampering.

## 3. AI Detection (Machine Learning)
Uses an XGBoost classification model and TF-IDF vectorization to analyze system call sequences and detect anomalous/malicious behavior.

- Train ML Model: Provide a path to a dataset containing system call sequences (normal vs. attack).
- Demo Mode: If no dataset is provided, the system will automatically generate a Demo Model using synthetic data so you can test the AI features immediately.

- Load ML Model: Loads a previously trained model (ml_intrusion_model.pkl) from memory.
- Test ML Detection: Runs a test sequence through the loaded model to verify anomaly detection.
- Show Training Report: Displays the model's Precision, Recall, and F1-Score metrics.
## 4. Security Alerts
- Real-Time Alerts: Automatically pops up desktop notifications and switches tabs for Critical and High severity events.
- Alert Management: View all triggered alerts (File Integrity violations or ML anomalies), clear the alert log, or export them to a .json file for auditing.
## 5. System Logs
A dedicated scrolling console that tracks all background activities, model training progress, file hashing events, and system errors in real-time.
## 6. Settings
- Hash Algorithm: Change the cryptographic algorithm used for file integrity (Default: SHA-256). 
- Check Interval: Adjust the frequency (in seconds) for periodic background integrity scans.
- Alert Threshold: Set the minimum severity level required to trigger desktop notifications and UI alerts.
-  Save Settings: Persists your configuration to hids_config.json.
## Project Structure

```text
PyHIDS/
├── hids_app.py              # Main application entry point and GUI orchestration
├── hids_config.json         # Default and user-defined security configurations
├── baseline.json            # Cryptographic baseline of monitored files (SHA-256, size, permissions)
├── ml_intrusion_model.pkl   # Serialized trained XGBoost model
├── ml_vectorizer.pkl        # Serialised CountVectorizer for feature extraction
├── security_alerts.log      # Persistent log of all generated security alerts
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
