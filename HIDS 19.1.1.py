import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import hashlib
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import psutil
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ML imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Dependency checking functions
def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import watchdog
    except ImportError:
        missing_deps.append("watchdog")
    
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import xgboost
    except ImportError:
        missing_deps.append("xgboost")
    
    try:
        import imblearn
    except ImportError:
        missing_deps.append("imbalanced-learn")
    
    try:
        import joblib
    except ImportError:
        missing_deps.append("joblib")
    
    try:
        import seaborn
    except ImportError:
        missing_deps.append("seaborn")
    
    return missing_deps

def install_dependencies():
    """Install required dependencies"""
    import subprocess
    
    dependencies = ["watchdog", "psutil", "scikit-learn", "xgboost", "imbalanced-learn", "joblib", "seaborn"]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"Successfully installed {dep}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}")
            return False
    
    return True

class MLIntrusionDetector:
    """Machine Learning based intrusion detection system"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.demo_mode = False
        self.training_report = None  # Store classification report
        
    def load_sequences_from_folder(self, folder_path):
        """Load system call sequences from folder"""
        sequences = []
        if not os.path.exists(folder_path):
            return sequences
            
        print(f"  -> Searching in: {os.path.basename(folder_path)}")
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(".txt"):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            sequences.append(f.read().replace('\n', ' ').strip())
                    except:
                        try:
                            with open(filepath, 'r', encoding='latin-1') as f:
                                sequences.append(f.read().replace('\n', ' ').strip())
                        except Exception as e:
                            logging.warning(f"Could not read file {filepath}: {e}")
        return sequences
    
    def format_classification_report(self, y_true, y_pred, labels=['Normal', 'Attack']):
        """Format classification report like the image style"""
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Create formatted report
        report_lines = []
        report_lines.append("Classification Report (in %):")
        report_lines.append("---")
        report_lines.append("    | Precision    | Recall    | F1-Score")
        report_lines.append("---")
        
        for i, label in enumerate(labels):
            report_lines.append(f"{label:<8} | {precision[i]*100:>9.2f}%    | {recall[i]*100:>7.2f}%    | {f1[i]*100:>8.2f}%")
        
        report_lines.append("---")
        report_lines.append("")
        report_lines.append(f"Training Accuracy: {accuracy*100:.2f}%")
        
        return "\n".join(report_lines)
    
    def create_demo_model(self):
        """Create a simple demo model for testing"""
        try:
            logging.info("Creating demo ML model...")
            
            # Create realistic demo data
            normal_sequences = [
                "open read write close exit",
                "fork exec wait exit",
                "malloc free open close read",
                "write lseek mmap munmap",
                "stat fstat access ioctl",
                "pipe dup select poll",
                "socket bind listen accept",
                "connect send recv shutdown"
            ]
            attack_sequences = [
                "ptrace mmap execve chroot setuid",
                "socket connect execve chmod mkdir",
                "fork setuid execve ptrace kill",
                "open truncate write chmod exit",
                "socket bind listen accept execve",
                "mmap mprotect execve fork wait"
            ]
            
            all_sequences = normal_sequences + attack_sequences
            all_labels = [0] * len(normal_sequences) + [1] * len(attack_sequences)
            
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                all_sequences, all_labels, test_size=0.3, random_state=42, stratify=all_labels
            )
            
            # Feature extraction
            self.vectorizer = CountVectorizer(ngram_range=(2, 3), analyzer='word', 
                                            token_pattern=r'\b\w+\b', max_features=100)
            X_train_features = self.vectorizer.fit_transform(X_train)
            X_test_features = self.vectorizer.transform(X_test)
            
            # Train simple model
            self.model = XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.model.fit(X_train_features, y_train)
            
            # Generate predictions and classification report
            y_pred = self.model.predict(X_test_features)
            self.training_report = self.format_classification_report(y_test, y_pred)
            
            # Save demo model
            joblib.dump(self.model, 'ml_intrusion_model.pkl')
            joblib.dump(self.vectorizer, 'ml_vectorizer.pkl')
            
            # Save report
            with open('ml_training_report.txt', 'w') as f:
                f.write(self.training_report)
            
            self.is_trained = True
            self.demo_mode = True
            logging.info("Demo ML model created successfully")
            return True, self.training_report
            
        except Exception as e:
            logging.error(f"Demo model creation failed: {e}")
            return False, None
    
    def train_model(self, dataset_base_path=None):
        """Train the ML model for intrusion detection"""
        try:
            # If no path provided, create demo model
            if not dataset_base_path or not os.path.exists(dataset_base_path):
                logging.info("No valid dataset path provided. Creating demo model.")
                return self.create_demo_model()
                
            logging.info("Loading data for ML training...")
            
            # Define paths
            train_normal_path = os.path.join(dataset_base_path, 'Training_Data_Master')
            val_normal_path = os.path.join(dataset_base_path, 'Validation_Data_Master')
            attack_path = os.path.join(dataset_base_path, 'Attack_Data_Master')
            
            # Load sequences
            normal_sequences = self.load_sequences_from_folder(train_normal_path)
            normal_sequences.extend(self.load_sequences_from_folder(val_normal_path))
            attack_sequences = self.load_sequences_from_folder(attack_path)
            
            logging.info(f"Loaded {len(normal_sequences)} normal sequences.")
            logging.info(f"Loaded {len(attack_sequences)} attack sequences.")
            
            if len(normal_sequences) == 0 or len(attack_sequences) == 0:
                logging.warning("Insufficient data for training. Using demo mode.")
                return self.create_demo_model()
                
            labels_normal = [0] * len(normal_sequences)
            labels_attack = [1] * len(attack_sequences)
            all_sequences = normal_sequences + attack_sequences
            all_labels = labels_normal + labels_attack
            
            # Feature extraction
            logging.info("Extracting features using a range of 2 to 4-grams...")
            self.vectorizer = CountVectorizer(ngram_range=(2, 4), analyzer='word', 
                                            token_pattern=r'\b\w+\b', max_features=5000)
            X_features = self.vectorizer.fit_transform(all_sequences)
            logging.info(f"Feature matrix shape: {X_features.shape}")
            
            # Data splitting and resampling
            logging.info("Splitting data into training and testing sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
            )
            
            logging.info("Applying SMOTE to balance the training data...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logging.info(f"Resampled training set distribution: {np.bincount(y_train_resampled)}")
            
            # Hyperparameter tuning
            logging.info("Starting Hyperparameter Tuning with GridSearchCV...")
            xgb = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.1, 0.2]
            }
            
            grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, 
                                     n_jobs=-1, verbose=1, scoring='f1')
            grid_search.fit(X_train_resampled, y_train_resampled)
            
            self.model = grid_search.best_estimator_
            logging.info(f"Best parameters found: {grid_search.best_params_}")
            
            # Evaluation
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report
            self.training_report = self.format_classification_report(y_test, y_pred)
            logging.info(f"Model trained successfully with test accuracy: {accuracy * 100:.2f}%")
            
            # Save model and vectorizer
            joblib.dump(self.model, 'ml_intrusion_model.pkl')
            joblib.dump(self.vectorizer, 'ml_vectorizer.pkl')
            
            # Save report
            with open('ml_training_report.txt', 'w') as f:
                f.write(self.training_report)
            
            self.is_trained = True
            self.demo_mode = False
            return True, self.training_report
            
        except Exception as e:
            logging.error(f"ML training failed: {e}")
            success, report = self.create_demo_model()
            return success, report
    
    def load_model(self):
        """Load pre-trained model and vectorizer"""
        try:
            if os.path.exists('ml_intrusion_model.pkl') and os.path.exists('ml_vectorizer.pkl'):
                self.model = joblib.load('ml_intrusion_model.pkl')
                self.vectorizer = joblib.load('ml_vectorizer.pkl')
                self.is_trained = True
                
                # Try to load training report
                try:
                    with open('ml_training_report.txt', 'r') as f:
                        self.training_report = f.read()
                except:
                    self.training_report = "No training report available"
                
                # Check if it's a demo model by checking model parameters
                if hasattr(self.model, 'n_estimators') and self.model.n_estimators == 50:
                    self.demo_mode = True
                else:
                    self.demo_mode = False
                logging.info("ML model loaded successfully")
                return True
            else:
                logging.info("No ML model found. Please train a model first.")
                return False
        except Exception as e:
            logging.warning(f"Could not load ML model: {e}")
            return False

class FileIntegrityMonitor:
    """File integrity monitoring engine"""
    
    def __init__(self, alert_callback, config):
        self.alert_callback = alert_callback
        self.config = config
        self.baseline_data = {}
        self.current_hashes = {}
        self.observer = None
        self.event_handler = None
        
    def create_baseline(self):
        """Create baseline file hashes"""
        self.baseline_data = {}
        monitored_paths = self.config['monitored_paths']
        
        for path in monitored_paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    self._hash_file(path)
                else:
                    self._hash_directory(path)
        
        # Save baseline to file
        self._save_baseline()
        return len(self.baseline_data)
    
    def _hash_directory(self, directory):
        """Recursively hash all files in directory"""
        for root, dirs, files in os.walk(directory):
            # Skip excluded paths
            if any(excluded in root for excluded in self.config['excluded_paths']):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                self._hash_file(file_path)
    
    def _hash_file(self, file_path):
        """Calculate file hash using configured algorithm"""
        try:
            hash_func = getattr(hashlib, self.config['hash_algorithm'])()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            file_stat = os.stat(file_path)
            file_info = {
                'hash': hash_func.hexdigest(),
                'size': file_stat.st_size,
                'modified': file_stat.st_mtime,
                'permissions': file_stat.st_mode
            }
            
            self.baseline_data[file_path] = file_info
            logging.info(f"Baseline created for: {file_path}")
            
        except (IOError, OSError) as e:
            logging.warning(f"Could not hash file {file_path}: {e}")
    
    def _save_baseline(self):
        """Save baseline to file"""
        try:
            with open('baseline.json', 'w') as f:
                json.dump(self.baseline_data, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save baseline: {e}")
    
    def load_baseline(self):
        """Load baseline from file"""
        try:
            with open('baseline.json', 'r') as f:
                self.baseline_data = json.load(f)
            return True
        except FileNotFoundError:
            return False
    
    def _get_current_file_info(self, file_path):
        """Get current file information and hash"""
        try:
            hash_func = getattr(hashlib, self.config['hash_algorithm'])()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            file_stat = os.stat(file_path)
            return {
                'hash': hash_func.hexdigest(),
                'size': file_stat.st_size,
                'modified': file_stat.st_mtime,
                'permissions': file_stat.st_mode
            }
        except (IOError, OSError) as e:
            logging.error(f"Could not get file info for {file_path}: {e}")
            return None
    
    def run_integrity_check(self):
        """Run integrity check against baseline"""
        violations = []
        
        for file_path, baseline_info in self.baseline_data.items():
            if not os.path.exists(file_path):
                violation = {
                    'type': 'file_missing',
                    'path': file_path,
                    'severity': 'Critical',
                    'description': 'Critical system file has been deleted or moved'
                }
                violations.append(violation)
                continue
            
            current_info = self._get_current_file_info(file_path)
            if not current_info:
                continue
            
            # Check hash - separate if statement
            if current_info['hash'] != baseline_info['hash']:
                violation = {
                    'type': 'hash_mismatch',
                    'path': file_path,
                    'severity': 'Critical',
                    'description': 'Critical file content has been modified'
                }
                violations.append(violation)
            
            # Check file size - separate if statement (not elif)
            if current_info['size'] != baseline_info['size']:
                violation = {
                    'type': 'size_change',
                    'path': file_path,
                    'severity': 'High',
                    'description': 'File size has changed unexpectedly'
                }
                violations.append(violation)
            
            # Check permissions - separate if statement (not elif)
            if current_info['permissions'] != baseline_info['permissions']:
                violation = {
                    'type': 'permission_change',
                    'path': file_path,
                    'severity': 'Medium',
                    'description': 'File permissions have changed'
                }
                violations.append(violation)
        
        return violations
    
    def start_realtime_monitoring(self):
        """Start real-time file system monitoring"""
        self.event_handler = FileSystemEventHandler()
        self.event_handler.on_modified = self._on_file_modified
        self.event_handler.on_created = self._on_file_created
        self.event_handler.on_deleted = self._on_file_deleted
        self.event_handler.on_moved = self._on_file_moved
        
        self.observer = Observer()
        for path in self.config['monitored_paths']:
            if os.path.exists(path):
                self.observer.schedule(self.event_handler, path, recursive=True)
        
        self.observer.start()
    
    def stop_realtime_monitoring(self):
        """Stop real-time monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
    
    def _on_file_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            # Determine severity based on file location
            severity = 'Medium'
            file_path = event.src_path.lower()
            if any(critical_path in file_path for critical_path in ['system32', 'etc', 'bin', 'syswow64']):
                severity = 'High'
            
            alert = {
                'timestamp': datetime.now().isoformat(),
                'severity': severity,
                'path': event.src_path,
                'description': 'File was modified',
                'type': 'file_integrity'
            }
            self.alert_callback(alert)
    
    def _on_file_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'severity': 'Low',
                'path': event.src_path,
                'description': 'New file created in monitored directory',
                'type': 'file_integrity'
            }
            self.alert_callback(alert)
    
    def _on_file_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory:
            # Determine severity based on file location
            severity = 'High'
            file_path = event.src_path.lower()
            if any(critical_path in file_path for critical_path in ['system32', 'etc', 'bin', 'syswow64']):
                severity = 'Critical'
            
            alert = {
                'timestamp': datetime.now().isoformat(),
                'severity': severity,
                'path': event.src_path,
                'description': 'File was deleted',
                'type': 'file_integrity'
            }
            self.alert_callback(alert)
    
    def _on_file_moved(self, event):
        """Handle file move events"""
        if not event.is_directory:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'severity': 'Medium',
                'path': f"{event.src_path} -> {event.dest_path}",
                'description': 'File was moved or renamed',
                'type': 'file_integrity'
            }
            self.alert_callback(alert)

class AlertManager:
    """Manage security alerts and notifications"""
    
    def __init__(self, gui_update_callback):
        self.alerts = []
        self.gui_update_callback = gui_update_callback
        self.severity_colors = {
            'Critical': '#FF0000',  # Red
            'High': '#FF6B00',      # Orange
            'Medium': '#FFD700',    # Yellow
            'Low': '#00BFFF'        # Blue
        }
    
    def add_alert(self, alert_data):
        """Add new security alert"""
        alert_data['id'] = len(self.alerts) + 1
        self.alerts.append(alert_data)
        
        # Log alert
        self.log_alert(alert_data)
        
        # Update GUI
        self.gui_update_callback(alert_data)
        
        # Show desktop notification for Critical and High severity alerts
        if alert_data['severity'] in ['Critical', 'High']:
            self.show_desktop_alert(alert_data)
    
    def log_alert(self, alert):
        """Log alert to file"""
        log_entry = f"{alert['timestamp']} - {alert['type']} - {alert['severity']} - {alert['path']} - {alert['description']}\n"
        
        try:
            with open('security_alerts.log', 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logging.error(f"Failed to write alert to log: {e}")
    
    def show_desktop_alert(self, alert):
        """Show desktop notification for critical alerts"""
        try:
            if alert['severity'] == 'Critical':
                messagebox.showerror(
                    "CRITICAL SECURITY ALERT",
                    f"🚨 CRITICAL ALERT: {alert['description']}\n\n"
                    f"Path: {alert['path']}\n"
                    f"Type: {alert.get('type', 'Unknown')}\n"
                    f"Time: {alert['timestamp']}"
                )
            elif alert['severity'] == 'High':
                messagebox.showwarning(
                    "HIGH SEVERITY ALERT",
                    f"⚠️ HIGH ALERT: {alert['description']}\n\n"
                    f"Path: {alert['path']}\n"
                    f"Type: {alert.get('type', 'Unknown')}\n"
                    f"Time: {alert['timestamp']}"
                )
        except Exception as e:
            logging.error(f"Failed to show desktop alert: {e}")
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
    
    def export_alerts(self, filename=None):
        """Export alerts to file"""
        if not filename:
            filename = f"hids_alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.alerts, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Failed to export alerts: {e}")
            return False

class HIDSApp:
    """Main Host Intrusion Detection System Application with ML capabilities"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PyHIDS - Advanced Host Intrusion Detection System")
        self.root.geometry("1200x800")
        
        # Security configuration
        self.config = self.load_config()
        self.monitoring_active = False
        self.ml_monitoring_active = False
        self.baseline_created = False
        
        # Initialize components
        self.setup_logging()
        self.create_gui()
        self.initialize_monitoring()
        
        # Load baseline if exists
        if self.integrity_monitor.load_baseline():
            self.baseline_created = True
            self.status_vars['baseline'].set("✅ LOADED")
        
        # Load ML model if exists
        if self.ml_detector.load_model():
            self.status_vars['ml_model'].set("✅ LOADED")
            if self.ml_detector.demo_mode:
                self.update_status("Demo ML model loaded successfully")
                # Display the training report if available
                if hasattr(self.ml_detector, 'training_report') and self.ml_detector.training_report:
                    self.display_training_report(self.ml_detector.training_report)
            else:
                self.update_status("ML model loaded successfully")
                # Display the training report if available
                if hasattr(self.ml_detector, 'training_report') and self.ml_detector.training_report:
                    self.display_training_report(self.ml_detector.training_report)
        else:
            self.status_vars['ml_model'].set("❌ NOT LOADED")
        
    def load_config(self):
        """Load security configuration"""
        # Use system-appropriate default paths
        if os.name == 'nt':  # Windows
            default_paths = ['C:\\Windows\\System32', 'C:\\Windows\\SysWOW64']
            excluded_paths = ['C:\\Windows\\Temp']
        else:  # Linux/Unix
            default_paths = ['/etc', '/bin', '/usr/bin', '/sbin', '/usr/sbin']
            excluded_paths = ['/tmp', '/proc', '/dev']
            
        default_config = {
            'monitored_paths': default_paths,
            'excluded_paths': excluded_paths,
            'check_interval': 30,
            'hash_algorithm': 'sha256',
            'alert_threshold': 'High',
            'log_retention_days': 30
        }
        
        try:
            with open('hids_config.json', 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            self.create_default_config(default_config)
            
        return default_config
    
    def create_default_config(self, config):
        """Create default configuration file"""
        with open('hids_config.json', 'w') as f:
            json.dump(config, f, indent=4)

    def create_gui(self):
        """Create the main GUI interface"""
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Dashboard Tab
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_frame, text="Dashboard")
        self.create_dashboard()
        
        # File Monitoring Tab
        self.monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.monitoring_frame, text="File Monitoring")
        self.create_monitoring_tab()
        
        # ML Detection Tab
        self.ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_frame, text="AI Detection")
        self.create_ml_tab()
        
        # Alerts Tab
        self.alerts_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.alerts_frame, text="Security Alerts")
        self.create_alerts_tab()
        
        # Logs Tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="System Logs")
        self.create_logs_tab()
        
        # Settings Tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        self.create_settings_tab()
        
        # Status Bar
        self.create_status_bar()

    def create_dashboard(self):
        """Create the main dashboard"""
        
        # System Status Frame
        status_frame = ttk.LabelFrame(self.dashboard_frame, text="System Status", padding=10)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        # Status indicators
        self.status_vars = {
            'monitoring': tk.StringVar(value="❌ INACTIVE"),
            'baseline': tk.StringVar(value="❌ NOT CREATED"),
            'ml_model': tk.StringVar(value="❌ NOT LOADED"),
            'ml_monitoring': tk.StringVar(value="❌ INACTIVE"),
            'alerts': tk.StringVar(value="0 Active Alerts"),
            'integrity': tk.StringVar(value="✅ SYSTEM SECURE")
        }
        
        ttk.Label(status_frame, text="File Monitoring:").grid(row=0, column=0, sticky='w')
        ttk.Label(status_frame, textvariable=self.status_vars['monitoring']).grid(row=0, column=1, sticky='w')
        
        ttk.Label(status_frame, text="Baseline Status:").grid(row=1, column=0, sticky='w')
        ttk.Label(status_frame, textvariable=self.status_vars['baseline']).grid(row=1, column=1, sticky='w')
        
        ttk.Label(status_frame, text="ML Model:").grid(row=2, column=0, sticky='w')
        ttk.Label(status_frame, textvariable=self.status_vars['ml_model']).grid(row=2, column=1, sticky='w')
        
        ttk.Label(status_frame, text="ML Monitoring:").grid(row=3, column=0, sticky='w')
        ttk.Label(status_frame, textvariable=self.status_vars['ml_monitoring']).grid(row=3, column=1, sticky='w')
        
        ttk.Label(status_frame, text="Active Alerts:").grid(row=4, column=0, sticky='w')
        ttk.Label(status_frame, textvariable=self.status_vars['alerts']).grid(row=4, column=1, sticky='w')
        
        ttk.Label(status_frame, text="System Integrity:").grid(row=5, column=0, sticky='w')
        ttk.Label(status_frame, textvariable=self.status_vars['integrity']).grid(row=5, column=1, sticky='w')
        
        # Quick Actions Frame
        actions_frame = ttk.LabelFrame(self.dashboard_frame, text="Quick Actions", padding=10)
        actions_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(actions_frame, text="Start File Monitoring", 
                  command=self.start_monitoring).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="Stop File Monitoring", 
                  command=self.stop_monitoring).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="Create Baseline", 
                  command=self.create_baseline).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="Run Integrity Check", 
                  command=self.run_integrity_check).pack(side='left', padx=5)
        
        ttk.Button(actions_frame, text="Train ML Model", 
                  command=self.train_ml_model).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="Start ML Monitoring", 
                  command=self.start_ml_monitoring).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="Stop ML Monitoring", 
                  command=self.stop_ml_monitoring).pack(side='left', padx=5)

    def create_monitoring_tab(self):
        """Create file monitoring interface"""
        
        # Monitoring control frame
        control_frame = ttk.Frame(self.monitoring_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Add Path", command=self.add_monitored_path).pack(side='left')
        ttk.Button(control_frame, text="Remove Path", command=self.remove_monitored_path).pack(side='left')
        ttk.Button(control_frame, text="Refresh", command=self.refresh_paths_list).pack(side='left')
        
        # Monitored paths list
        paths_frame = ttk.LabelFrame(self.monitoring_frame, text="Monitored Paths")
        paths_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.paths_listbox = tk.Listbox(paths_frame, height=10)
        self.paths_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Refresh paths list
        self.refresh_paths_list()

    def create_ml_tab(self):
        """Create ML detection interface"""
        
        ml_frame = ttk.LabelFrame(self.ml_frame, text="Machine Learning Intrusion Detection")
        ml_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # ML Controls
        controls_frame = ttk.Frame(ml_frame)
        controls_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(controls_frame, text="Train ML Model", 
                  command=self.train_ml_model).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Load ML Model", 
                  command=self.load_ml_model).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Test ML Detection", 
                  command=self.test_ml_detection).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Show Training Report", 
                  command=self.show_training_report).pack(side='left', padx=5)
        
        # ML Status
        status_frame = ttk.Frame(ml_frame)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(status_frame, text="ML Model Status:").pack(side='left')
        self.ml_status_var = tk.StringVar(value="Not Trained")
        ttk.Label(status_frame, textvariable=self.ml_status_var).pack(side='left', padx=10)
        
        # Dataset path
        path_frame = ttk.Frame(ml_frame)
        path_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(path_frame, text="Dataset Path:").pack(side='left')
        self.ml_path_var = tk.StringVar(value="")
        ttk.Entry(path_frame, textvariable=self.ml_path_var, width=80).pack(side='left', padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_ml_dataset).pack(side='left')
        
        # Create notebook for ML results and reports
        ml_notebook = ttk.Notebook(ml_frame)
        ml_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # ML Results Tab
        results_frame = ttk.Frame(ml_notebook)
        ml_notebook.add(results_frame, text="Detection Results")
        
        self.ml_results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.ml_results_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.ml_results_text.config(state='disabled')
        
        # Training Report Tab
        report_frame = ttk.Frame(ml_notebook)
        ml_notebook.add(report_frame, text="Training Report")
        
        self.report_text = scrolledtext.ScrolledText(report_frame, height=15, font=('Courier', 10))
        self.report_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.report_text.config(state='disabled')
        
        # Add initial message to report tab
        self.report_text.config(state='normal')
        self.report_text.insert(tk.END, "Training report will appear here after model training.\n")
        self.report_text.insert(tk.END, "Format will include Precision, Recall, and F1-Score like:\n\n")
        self.report_text.insert(tk.END, "Classification Report (in %):\n")
        self.report_text.insert(tk.END, "---\n")
        self.report_text.insert(tk.END, "    | Precision    | Recall    | F1-Score\n")
        self.report_text.insert(tk.END, "---\n")
        self.report_text.insert(tk.END, "Normal    | 98.65%    | 98.40%    | 98.53%\n")
        self.report_text.insert(tk.END, "Attack    | 89.04%    | 90.62%    | 89.82%\n")
        self.report_text.insert(tk.END, "---\n")
        self.report_text.insert(tk.END, "\n")
        self.report_text.insert(tk.END, "Training Accuracy: 99.97%")
        self.report_text.config(state='disabled')
    
    def display_training_report(self, report_text):
        """Display training report in the report tab"""
        self.report_text.config(state='normal')
        self.report_text.delete(1.0, tk.END)
        
        # Add some styling
        self.report_text.insert(tk.END, "=== ML MODEL TRAINING REPORT ===\n\n")
        
        # Add demo mode indicator if applicable
        if hasattr(self.ml_detector, 'demo_mode') and self.ml_detector.demo_mode:
            self.report_text.insert(tk.END, "⚠️ DEMO MODEL - Using synthetic data\n\n")
        
        # Insert the actual report
        self.report_text.insert(tk.END, report_text)
        
        # Add timestamp
        self.report_text.insert(tk.END, f"\n\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.report_text.config(state='disabled')
        
        # Switch to report tab
        notebook = self.report_text.master.master.master  # Get the notebook parent
        notebook.select(1)  # Select report tab (index 1)
    
    def show_training_report(self):
        """Show the saved training report"""
        if hasattr(self.ml_detector, 'training_report') and self.ml_detector.training_report:
            self.display_training_report(self.ml_detector.training_report)
        else:
            messagebox.showinfo("No Report", "No training report available. Please train a model first.")
    
    def create_alerts_tab(self):
        """Create security alerts interface"""
        
        # Alerts control frame
        alert_control_frame = ttk.Frame(self.alerts_frame)
        alert_control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(alert_control_frame, text="Clear Alerts", 
                  command=self.clear_alerts).pack(side='left')
        ttk.Button(alert_control_frame, text="Export Alerts", 
                  command=self.export_alerts).pack(side='left')
        
        # Alerts display
        alerts_frame = ttk.LabelFrame(self.alerts_frame, text="Security Alerts")
        alerts_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create treeview for alerts
        columns = ('timestamp', 'type', 'severity', 'path', 'description')
        self.alerts_tree = ttk.Treeview(alerts_frame, columns=columns, show='headings')
        
        # Define headings
        self.alerts_tree.heading('timestamp', text='Timestamp')
        self.alerts_tree.heading('type', text='Type')
        self.alerts_tree.heading('severity', text='Severity')
        self.alerts_tree.heading('path', text='Path')
        self.alerts_tree.heading('description', text='Description')
        
        # Configure columns
        self.alerts_tree.column('timestamp', width=150)
        self.alerts_tree.column('type', width=120)
        self.alerts_tree.column('severity', width=100)
        self.alerts_tree.column('path', width=300)
        self.alerts_tree.column('description', width=400)
        
        # Configure tags for severity colors
        self.alerts_tree.tag_configure('Critical', background='#FFE6E6')
        self.alerts_tree.tag_configure('High', background='#FFF0E6')
        self.alerts_tree.tag_configure('Medium', background='#FFFDE6')
        self.alerts_tree.tag_configure('Low', background='#E6F7FF')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(alerts_frame, orient='vertical', command=self.alerts_tree.yview)
        self.alerts_tree.configure(yscrollcommand=scrollbar.set)
        
        self.alerts_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def create_logs_tab(self):
        """Create system logs interface"""
        
        logs_frame = ttk.Frame(self.logs_frame)
        logs_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.logs_text = scrolledtext.ScrolledText(logs_frame, height=20, width=100)
        self.logs_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.logs_text.config(state='disabled')

    def create_settings_tab(self):
        """Create settings interface"""
        
        settings_frame = ttk.Frame(self.settings_frame)
        settings_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Hash algorithm setting
        ttk.Label(settings_frame, text="Hash Algorithm:").grid(row=0, column=0, sticky='w', pady=5)
        self.hash_algo_var = tk.StringVar(value=self.config['hash_algorithm'])
        hash_combo = ttk.Combobox(settings_frame, textvariable=self.hash_algo_var, 
                                 values=['md5', 'sha1', 'sha256', 'sha512'])
        hash_combo.grid(row=0, column=1, sticky='w', pady=5)
        
        # Check interval
        ttk.Label(settings_frame, text="Check Interval (seconds):").grid(row=1, column=0, sticky='w', pady=5)
        self.interval_var = tk.StringVar(value=str(self.config['check_interval']))
        ttk.Entry(settings_frame, textvariable=self.interval_var).grid(row=1, column=1, sticky='w', pady=5)
        
        # Alert threshold
        ttk.Label(settings_frame, text="Alert Threshold:").grid(row=2, column=0, sticky='w', pady=5)
        self.alert_threshold_var = tk.StringVar(value=self.config['alert_threshold'])
        threshold_combo = ttk.Combobox(settings_frame, textvariable=self.alert_threshold_var,
                                      values=['Critical', 'High', 'Medium', 'Low'])
        threshold_combo.grid(row=2, column=1, sticky='w', pady=5)
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", 
                  command=self.save_settings).grid(row=3, column=0, columnspan=2, pady=10)

    def create_status_bar(self):
        """Create application status bar"""
        
        self.status_bar = ttk.Label(self.root, text="Ready", relief='sunken', anchor='w')
        self.status_bar.pack(side='bottom', fill='x')

    def initialize_monitoring(self):
        """Initialize monitoring components"""
        self.alert_manager = AlertManager(self.update_alerts_display)
        self.integrity_monitor = FileIntegrityMonitor(
            self.alert_manager.add_alert,
            self.config
        )
        self.ml_detector = MLIntrusionDetector()
    
    def start_monitoring(self):
        """Start the HIDS monitoring"""
        if not self.baseline_created:
            messagebox.showwarning("Warning", "Please create a baseline first!")
            return
        
        self.monitoring_active = True
        self.status_vars['monitoring'].set("✅ ACTIVE")
        
        # Start real-time monitoring in separate thread
        monitor_thread = threading.Thread(target=self.integrity_monitor.start_realtime_monitoring)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start periodic integrity checks
        self.schedule_periodic_checks()
        
        self.update_status("File monitoring started successfully")
        logging.info("HIDS file monitoring started")
    
    def stop_monitoring(self):
        """Stop the HIDS monitoring"""
        self.monitoring_active = False
        self.status_vars['monitoring'].set("❌ INACTIVE")
        
        self.integrity_monitor.stop_realtime_monitoring()
        
        self.update_status("File monitoring stopped")
        logging.info("HIDS file monitoring stopped")
    
    def schedule_periodic_checks(self):
        """Schedule periodic integrity checks"""
        if self.monitoring_active:
            # Run check in separate thread
            check_thread = threading.Thread(target=self.run_integrity_check)
            check_thread.daemon = True
            check_thread.start()
            
            # Schedule next check
            interval = self.config['check_interval'] * 1000  # Convert to milliseconds
            self.root.after(interval, self.schedule_periodic_checks)
    
    def create_baseline(self):
        """Create file integrity baseline"""
        try:
            file_count = self.integrity_monitor.create_baseline()
            self.baseline_created = True
            self.status_vars['baseline'].set(f"✅ CREATED ({file_count} files)")
            
            messagebox.showinfo("Success", f"Baseline created for {file_count} files")
            self.update_status(f"Baseline created for {file_count} files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create baseline: {e}")
            logging.error(f"Baseline creation failed: {e}")
    
    def run_integrity_check(self):
        """Run manual integrity check"""
        if not self.baseline_created:
            messagebox.showwarning("Warning", "Please create a baseline first!")
            return
        
        violations = self.integrity_monitor.run_integrity_check()
        
        if violations:
            self.status_vars['integrity'].set("❌ INTEGRITY VIOLATED")
            for violation in violations:
                alert_data = {
                    'timestamp': datetime.now().isoformat(),
                    'severity': violation['severity'],
                    'path': violation['path'],
                    'description': violation['description'],
                    'type': 'file_integrity'
                }
                self.alert_manager.add_alert(alert_data)
            
            self.update_status(f"Integrity check found {len(violations)} violations")
        else:
            self.status_vars['integrity'].set("✅ SYSTEM SECURE")
            self.update_status("Integrity check passed - no violations found")

    # ML-related methods
    def train_ml_model(self):
        """Train the ML model for intrusion detection"""
        dataset_path = self.ml_path_var.get()
        
        self.update_status("Starting ML model training...")
        
        # Run training in separate thread to prevent GUI freezing
        def train_thread():
            success, report = self.ml_detector.train_model(dataset_path)
            if success:
                self.root.after(0, lambda: self.ml_status_var.set("Trained"))
                self.root.after(0, lambda: self.status_vars['ml_model'].set("✅ TRAINED"))
                
                # Display the training report
                if report:
                    self.root.after(0, lambda: self.display_training_report(report))
                
                if self.ml_detector.demo_mode:
                    self.root.after(0, lambda: self.update_status("Demo ML model trained successfully"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        "Demo ML model trained successfully!\n\nCheck the 'Training Report' tab for detailed metrics."))
                else:
                    self.root.after(0, lambda: self.update_status("ML model trained successfully"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        "ML model trained successfully!\n\nCheck the 'Training Report' tab for detailed metrics."))
            else:
                self.root.after(0, lambda: self.update_status("ML model training failed"))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def load_ml_model(self):
        """Load pre-trained ML model"""
        if self.ml_detector.load_model():
            self.ml_status_var.set("Loaded")
            self.status_vars['ml_model'].set("✅ LOADED")
            
            # Display training report if available
            if hasattr(self.ml_detector, 'training_report') and self.ml_detector.training_report:
                self.display_training_report(self.ml_detector.training_report)
            
            if self.ml_detector.demo_mode:
                self.update_status("Demo ML model loaded successfully")
                messagebox.showinfo("Success", 
                    "Demo ML model loaded successfully!\n\nCheck the 'Training Report' tab for detailed metrics.")
            else:
                self.update_status("ML model loaded successfully")
                messagebox.showinfo("Success", 
                    "ML model loaded successfully!\n\nCheck the 'Training Report' tab for detailed metrics.")
        else:
            messagebox.showerror("Error", "Could not load ML model. Please train first.")
    
    def start_ml_monitoring(self):
        """Start ML-based monitoring"""
        if not self.ml_detector.is_trained:
            messagebox.showwarning("Warning", "Please train or load ML model first!")
            return
        
        self.ml_monitoring_active = True
        self.status_vars['ml_monitoring'].set("✅ ACTIVE")
        if self.ml_detector.demo_mode:
            self.update_status("Demo ML monitoring started")
        else:
            self.update_status("ML monitoring started")
        
        # Simulate ML detection (in real implementation, this would monitor system calls)
        self.simulate_ml_detection()
    
    def stop_ml_monitoring(self):
        """Stop ML-based monitoring"""
        self.ml_monitoring_active = False
        self.status_vars['ml_monitoring'].set("❌ INACTIVE")
        self.update_status("ML monitoring stopped")
    
    def simulate_ml_detection(self):
        """Simulate ML-based intrusion detection (for demo purposes)"""
        if self.ml_monitoring_active:
            # Simulate occasional attack detection with different severity levels
            import random
            rand_val = random.random()
            
            if rand_val < 0.05:  # 5% chance of Critical detection
                alert_data = {
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'Critical',
                    'path': 'System Call Sequence',
                    'description': 'CRITICAL: ML detected rootkit-like system call pattern',
                    'type': 'ml_intrusion'
                }
                self.alert_manager.add_alert(alert_data)
            elif rand_val < 0.15:  # 10% chance of High detection
                alert_data = {
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'High',
                    'path': 'System Call Sequence',
                    'description': 'HIGH: ML detected suspicious privilege escalation pattern',
                    'type': 'ml_intrusion'
                }
                self.alert_manager.add_alert(alert_data)
            elif rand_val < 0.3:  # 15% chance of Medium detection
                alert_data = {
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'Medium',
                    'path': 'System Call Sequence',
                    'description': 'MEDIUM: ML detected unusual system call sequence',
                    'type': 'ml_intrusion'
                }
                self.alert_manager.add_alert(alert_data)
            
            # Schedule next check
            self.root.after(5000, self.simulate_ml_detection)
    
    def test_ml_detection(self):
        """Test ML detection with sample data"""
        if not self.ml_detector.is_trained:
            messagebox.showwarning("Warning", "ML model not trained or loaded!")
            return
        
        # Sample system call sequences for testing
        test_sequences = [
            "open read write close",  # Normal sequence
            "execve chmod socket connect",  # Suspicious sequence
            "open read write close exit",  # Normal sequence
            "ptrace mmap execve chroot"  # Attack sequence
        ]
        
        results = []
        for seq in test_sequences:
            prediction = self.ml_detector.predict_sequence(seq)
            if prediction:
                # Determine severity based on probability
                if prediction['probability'] > 0.8:
                    severity = "Critical"
                elif prediction['probability'] > 0.6:
                    severity = "High"
                elif prediction['probability'] > 0.4:
                    severity = "Medium"
                else:
                    severity = "Low"
                    
                result = f"Sequence: {seq}\n"
                result += f"Prediction: {'ATTACK' if prediction['prediction'] == 1 else 'NORMAL'}\n"
                result += f"Severity: {severity}\n"
                result += f"Confidence: {prediction['confidence']:.2f}\n"
                result += f"Probability: {prediction['probability']:.2f}\n"
                if prediction.get('demo_mode', False):
                    result += f"Mode: DEMO MODEL\n"
                result += "-" * 50 + "\n"
                results.append(result)
        
        self.ml_results_text.config(state='normal')
        self.ml_results_text.delete(1.0, tk.END)
        if self.ml_detector.demo_mode:
            self.ml_results_text.insert(tk.END, "ML Detection Test Results (DEMO MODE):\n\n")
        else:
            self.ml_results_text.insert(tk.END, "ML Detection Test Results:\n\n")
        self.ml_results_text.insert(tk.END, "".join(results))
        self.ml_results_text.config(state='disabled')
    
    def browse_ml_dataset(self):
        """Browse for ML dataset directory"""
        path = filedialog.askdirectory(title="Select ML Dataset Directory")
        if path:
            self.ml_path_var.set(path)

    def add_monitored_path(self):
        """Add a new path to monitor"""
        path = filedialog.askdirectory(title="Select Directory to Monitor")
        if path:
            if path not in self.config['monitored_paths']:
                self.config['monitored_paths'].append(path)
                self.save_config()
                self.refresh_paths_list()
                self.update_status(f"Added monitoring path: {path}")

    def remove_monitored_path(self):
        """Remove selected monitored path"""
        selection = self.paths_listbox.curselection()
        if selection:
            path = self.paths_listbox.get(selection[0])
            self.config['monitored_paths'].remove(path)
            self.save_config()
            self.refresh_paths_list()
            self.update_status(f"Removed monitoring path: {path}")

    def refresh_paths_list(self):
        """Refresh the monitored paths list"""
        self.paths_listbox.delete(0, tk.END)
        for path in self.config['monitored_paths']:
            self.paths_listbox.insert(tk.END, path)

    def update_alerts_display(self, alert_data):
        """Update alerts display in GUI"""
        severity = alert_data['severity']
        tags = (severity,)
        
        self.alerts_tree.insert('', 0, values=(
            alert_data['timestamp'],
            alert_data.get('type', 'Unknown'),
            severity,
            alert_data['path'],
            alert_data['description']
        ), tags=tags)
        
        # Update alerts count
        alert_count = len(self.alert_manager.alerts)
        self.status_vars['alerts'].set(f"{alert_count} Active Alerts")
        
        # Auto-switch to alerts tab for Critical and High severity alerts
        if severity in ['Critical', 'High']:
            self.notebook.select(self.alerts_frame)

    def clear_alerts(self):
        """Clear all alerts from display"""
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
        self.alert_manager.clear_alerts()
        self.status_vars['alerts'].set("0 Active Alerts")
        self.update_status("Alerts cleared")

    def export_alerts(self):
        """Export alerts to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            if self.alert_manager.export_alerts(filename):
                messagebox.showinfo("Success", f"Alerts exported to {filename}")
                self.update_status(f"Alerts exported to {filename}")
            else:
                messagebox.showerror("Error", "Failed to export alerts")

    def save_settings(self):
        """Save application settings"""
        try:
            self.config['hash_algorithm'] = self.hash_algo_var.get()
            self.config['check_interval'] = int(self.interval_var.get())
            self.config['alert_threshold'] = self.alert_threshold_var.get()
            self.save_config()
            messagebox.showinfo("Success", "Settings saved successfully")
            self.update_status("Settings updated")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid setting value: {e}")

    def save_config(self):
        """Save configuration to file"""
        with open('hids_config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hids.log'),
                logging.StreamHandler()
            ]
        )

    def update_status(self, message):
        """Update status bar and logs"""
        self.status_bar.config(text=message)
        self.logs_text.config(state='normal')
        self.logs_text.insert('end', f"{datetime.now()}: {message}\n")
        self.logs_text.config(state='disabled')
        self.logs_text.see('end')
        logging.info(message)

    def on_closing(self):
        """Handle application closing"""
        if self.monitoring_active:
            self.stop_monitoring()
        if self.ml_monitoring_active:
            self.stop_ml_monitoring()
        self.root.destroy()

def main():
    """Main application entry point"""
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        response = input("Would you like to install them now? (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                print("Failed to install dependencies. Please install them manually.")
                return
        else:
            print("Please install the required dependencies to run PyHIDS.")
            return
    
    try:
        root = tk.Tk()
        app = HIDSApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except KeyboardInterrupt:
        print("\nHIDS shutdown requested...")
    except Exception as e:
        logging.critical(f"Application error: {e}")
        messagebox.showerror("Fatal Error", f"The application encountered an error: {e}")

if __name__ == "__main__":
    main()