#  PyHIDS — Advanced Host Intrusion Detection System

PyHIDS is an advanced, hybrid **Host-based Intrusion Detection System (HIDS)** designed using Python, integrating signature-based **File Integrity Monitoring (FIM)** with anomaly-based **Machine Learning (ML)** detection.

Inspired by industry-standard tools like **OSSEC** and **Tripwire**, PyHIDS adds modern machine learning techniques to detect zero-day and anomalous system-call patterns. It demonstrates sophisticated techniques for real-time file system monitoring, cryptographic baseline creation, and behavioural analysis.

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
- `CountVectorizer` for **2–4 n-gram** feature extraction from system call logs.
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
- Prediction: Evaluates incoming sequences (e.g., ptrace mmap execve chroot). Severity based on probability thresholds (>0.8 = Critical, >0.6 = High).
- Demo Mode: Generates realistic demo data and trains a lightweight model on the fly — test the system immediately without external training data.

## AlertManager
- Aggregates alerts, assigns sequential IDs, updates the GUI.
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
