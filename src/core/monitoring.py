"""
Advanced monitoring and metrics system for ML pipeline.

This module provides:
- Real-time performance monitoring
- Custom metrics collection
- Health checks and alerts
- Resource usage tracking
- Performance profiling
"""

import time
import psutil
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import weakref

from .exceptions import MonitoringError
from .logger import Logger


@dataclass
class Metric:
    """A metric measurement."""
    name: str
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    function_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call: Optional[datetime] = None
    
    def add_call(self, duration: float) -> None:
        """Add a function call measurement."""
        self.total_calls += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.total_calls
        self.last_call = datetime.now()


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    @abstractmethod
    def collect(self) -> List[Metric]:
        """Collect metrics."""
        pass


class SystemMetricCollector(MetricCollector):
    """Collects system-level metrics."""
    
    def collect(self) -> List[Metric]:
        """Collect system metrics."""
        metrics = []
        now = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(Metric("system.cpu.usage", cpu_percent, now, unit="percent"))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(Metric("system.memory.usage", memory.percent, now, unit="percent"))
        metrics.append(Metric("system.memory.available", memory.available, now, unit="bytes"))
        metrics.append(Metric("system.memory.total", memory.total, now, unit="bytes"))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(Metric("system.disk.usage", disk.percent, now, unit="percent"))
        metrics.append(Metric("system.disk.free", disk.free, now, unit="bytes"))
        metrics.append(Metric("system.disk.total", disk.total, now, unit="bytes"))
        
        # Network metrics
        network = psutil.net_io_counters()
        metrics.append(Metric("system.network.bytes_sent", network.bytes_sent, now, unit="bytes"))
        metrics.append(Metric("system.network.bytes_recv", network.bytes_recv, now, unit="bytes"))
        
        return metrics


class MLMetricCollector(MetricCollector):
    """Collects ML-specific metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
    
    def add_metric(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Add a custom ML metric."""
        with self._lock:
            metric = Metric(name, value, tags=tags or {})
            self._metrics[name].append(metric)
    
    def collect(self) -> List[Metric]:
        """Collect ML metrics."""
        with self._lock:
            metrics = []
            for metric_list in self._metrics.values():
                metrics.extend(metric_list)
            return metrics


class HealthChecker(ABC):
    """Abstract base class for health checkers."""
    
    @abstractmethod
    def check(self) -> HealthCheck:
        """Perform health check."""
        pass


class SystemHealthChecker(HealthChecker):
    """System-level health checker."""
    
    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
    
    def check(self) -> HealthCheck:
        """Check system health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            issues = []
            status = "healthy"
            
            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                status = "unhealthy"
            elif cpu_percent > self.cpu_threshold * 0.8:
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
                status = "degraded"
            
            if memory.percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                status = "unhealthy"
            elif memory.percent > self.memory_threshold * 0.8:
                issues.append(f"Elevated memory usage: {memory.percent:.1f}%")
                status = "degraded"
            
            message = "; ".join(issues) if issues else "System is healthy"
            
            return HealthCheck(
                name="system",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available": memory.available,
                    "disk_usage": psutil.disk_usage('/').percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="system",
                status="unhealthy",
                message=f"Health check failed: {e}",
                details={"error": str(e)}
            )


class PerformanceProfiler:
    """Performance profiler for functions and methods."""
    
    def __init__(self):
        self._profiles: Dict[str, PerformanceProfile] = {}
        self._lock = threading.RLock()
    
    def profile(self, func: Callable) -> Callable:
        """Decorator to profile function performance."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                self._add_profile(func.__name__, duration)
        return wrapper
    
    def _add_profile(self, function_name: str, duration: float) -> None:
        """Add a profile measurement."""
        with self._lock:
            if function_name not in self._profiles:
                self._profiles[function_name] = PerformanceProfile(function_name)
            self._profiles[function_name].add_call(duration)
    
    def get_profiles(self) -> Dict[str, PerformanceProfile]:
        """Get all performance profiles."""
        with self._lock:
            return self._profiles.copy()
    
    def get_profile(self, function_name: str) -> Optional[PerformanceProfile]:
        """Get profile for a specific function."""
        with self._lock:
            return self._profiles.get(function_name)
    
    def clear_profiles(self) -> None:
        """Clear all profiles."""
        with self._lock:
            self._profiles.clear()


class MonitoringSystem:
    """Centralized monitoring system."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self._collectors: List[MetricCollector] = []
        self._health_checkers: List[HealthChecker] = []
        self._profiler = PerformanceProfiler()
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._alerts: List[Callable] = []
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Add default collectors and checkers
        self.add_collector(SystemMetricCollector())
        self.add_collector(MLMetricCollector())
        self.add_health_checker(SystemHealthChecker())
    
    def add_collector(self, collector: MetricCollector) -> None:
        """Add a metric collector."""
        with self._lock:
            self._collectors.append(collector)
    
    def add_health_checker(self, checker: HealthChecker) -> None:
        """Add a health checker."""
        with self._lock:
            self._health_checkers.append(checker)
    
    def add_alert(self, alert_func: Callable[[HealthCheck], None]) -> None:
        """Add an alert handler."""
        with self._lock:
            self._alerts.append(alert_func)
    
    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start the monitoring system."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
        if self.logger:
            self.logger.info(f"Monitoring system started with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        if self.logger:
            self.logger.info("Monitoring system stopped")
    
    def collect_metrics(self) -> List[Metric]:
        """Collect metrics from all collectors."""
        all_metrics = []
        with self._lock:
            for collector in self._collectors:
                try:
                    metrics = collector.collect()
                    all_metrics.extend(metrics)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error collecting metrics from {collector.__class__.__name__}: {e}")
        
        # Store metrics in history
        for metric in all_metrics:
            self._metrics_history[metric.name].append(metric)
        
        return all_metrics
    
    def perform_health_checks(self) -> List[HealthCheck]:
        """Perform all health checks."""
        health_checks = []
        with self._lock:
            for checker in self._health_checkers:
                try:
                    health_check = checker.check()
                    health_checks.append(health_check)
                    
                    # Trigger alerts for unhealthy checks
                    if health_check.status in ["unhealthy", "degraded"]:
                        for alert_func in self._alerts:
                            try:
                                alert_func(health_check)
                            except Exception as e:
                                if self.logger:
                                    self.logger.error(f"Error in alert handler: {e}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in health checker {checker.__class__.__name__}: {e}")
        
        return health_checks
    
    def get_metric_history(self, metric_name: str, duration: Optional[timedelta] = None) -> List[Metric]:
        """Get metric history for a specific metric."""
        with self._lock:
            if metric_name not in self._metrics_history:
                return []
            
            metrics = list(self._metrics_history[metric_name])
            
            if duration:
                cutoff_time = datetime.now() - duration
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            return metrics
    
    def get_metric_summary(self, metric_name: str, duration: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        metrics = self.get_metric_history(metric_name, duration)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "first_timestamp": metrics[0].timestamp,
            "last_timestamp": metrics[-1].timestamp
        }
    
    def get_performance_profiles(self) -> Dict[str, PerformanceProfile]:
        """Get all performance profiles."""
        return self._profiler.get_profiles()
    
    def profile_function(self, func: Callable) -> Callable:
        """Profile a function."""
        return self._profiler.profile(func)
    
    def export_metrics(self, filepath: str, duration: Optional[timedelta] = None) -> None:
        """Export metrics to a file."""
        all_metrics = []
        for metric_name in self._metrics_history:
            metrics = self.get_metric_history(metric_name, duration)
            all_metrics.extend(metrics)
        
        # Convert to JSON-serializable format
        export_data = []
        for metric in all_metrics:
            export_data.append({
                "name": metric.name,
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "tags": metric.tags,
                "unit": metric.unit
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Exported {len(export_data)} metrics to {filepath}")
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                if self.logger:
                    self.logger.debug(f"Collected {len(metrics)} metrics")
                
                # Perform health checks
                health_checks = self.perform_health_checks()
                unhealthy_checks = [hc for hc in health_checks if hc.status != "healthy"]
                if unhealthy_checks and self.logger:
                    self.logger.warning(f"Found {len(unhealthy_checks)} unhealthy health checks")
                
                time.sleep(interval)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)


# Global monitoring system instance
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> MonitoringSystem:
    """Get the global monitoring system."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system


def start_monitoring(interval: float = 30.0) -> None:
    """Start the global monitoring system."""
    get_monitoring_system().start_monitoring(interval)


def stop_monitoring() -> None:
    """Stop the global monitoring system."""
    global _monitoring_system
    if _monitoring_system:
        _monitoring_system.stop_monitoring()


def profile_function(func: Callable) -> Callable:
    """Profile a function using the global monitoring system."""
    return get_monitoring_system().profile_function(func)


def add_metric(name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
    """Add a custom metric to the global monitoring system."""
    monitoring_system = get_monitoring_system()
    for collector in monitoring_system._collectors:
        if isinstance(collector, MLMetricCollector):
            collector.add_metric(name, value, tags)
            break
