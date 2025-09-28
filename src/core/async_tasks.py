"""
Asynchronous task system for ML pipeline operations.

This module provides:
- Async task execution with progress tracking
- Task queuing and scheduling
- Resource management and throttling
- Task monitoring and failure recovery
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Awaitable
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref

from .exceptions import TaskError, TaskTimeoutError
from .logger import Logger


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start_time and self.end_time:
            self.duration = self.end_time - self.start_time


@dataclass
class TaskProgress:
    """Task execution progress information."""
    task_id: str
    current: int = 0
    total: int = 100
    percentage: float = 0.0
    message: str = ""
    stage: str = ""
    estimated_remaining: Optional[timedelta] = None
    
    def __post_init__(self):
        if self.total > 0:
            self.percentage = (self.current / self.total) * 100


class Task(ABC):
    """Abstract base class for tasks."""
    
    def __init__(
        self,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[timedelta] = None,
        retry_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.priority = priority
        self.timeout = timeout
        self.retry_count = retry_count
        self.metadata = metadata or {}
        self.status = TaskStatus.PENDING
        self.progress = TaskProgress(task_id=self.task_id)
        self.result: Optional[TaskResult] = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
    
    @abstractmethod
    async def execute(self) -> Any:
        """Execute the task asynchronously."""
        pass
    
    def update_progress(
        self, 
        current: int, 
        total: int = 100, 
        message: str = "", 
        stage: str = ""
    ) -> None:
        """Update task progress."""
        self.progress.current = current
        self.progress.total = total
        self.progress.message = message
        self.progress.stage = stage
        if total > 0:
            self.progress.percentage = (current / total) * 100


class FunctionTask(Task):
    """Task wrapper for regular functions."""
    
    def __init__(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        **task_kwargs
    ):
        super().__init__(**task_kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
    
    async def execute(self) -> Any:
        """Execute the wrapped function."""
        # Run in thread pool for non-async functions
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.func(*self.args, **self.kwargs)
        )


class AsyncFunctionTask(Task):
    """Task wrapper for async functions."""
    
    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        **task_kwargs
    ):
        super().__init__(**task_kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
    
    async def execute(self) -> Any:
        """Execute the wrapped async function."""
        return await self.func(*self.args, **self.kwargs)


class TaskScheduler:
    """Advanced task scheduler with priority queuing and resource management."""
    
    def __init__(
        self,
        max_workers: int = 4,
        max_memory_mb: int = 1024,
        logger: Optional[Logger] = None
    ):
        self.max_workers = max_workers
        self.max_memory_mb = max_memory_mb
        self.logger = logger
        
        # Task management
        self._tasks: Dict[str, Task] = {}
        self._task_queue = queue.PriorityQueue()
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        
        # Execution control
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'total_execution_time': timedelta()
        }
    
    async def start(self) -> None:
        """Start the task scheduler."""
        if self._running:
            return
        
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        if self.logger:
            self.logger.info(f"Task scheduler started with {self.max_workers} workers")
    
    async def stop(self) -> None:
        """Stop the task scheduler."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel running tasks
        for task_id, asyncio_task in self._running_tasks.items():
            asyncio_task.cancel()
            if self.logger:
                self.logger.info(f"Cancelled running task: {task_id}")
        
        # Wait for scheduler to stop
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        if self.logger:
            self.logger.info("Task scheduler stopped")
    
    def submit_task(self, task: Task) -> str:
        """Submit a task for execution."""
        with self._lock:
            self._tasks[task.task_id] = task
            self._task_queue.put((task.priority.value, task.created_at, task))
            self._stats['total_tasks'] += 1
            
            if self.logger:
                self.logger.debug(f"Submitted task: {task.task_id}")
            
            return task.task_id
    
    def submit_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        **task_kwargs
    ) -> str:
        """Submit a function as a task."""
        task = FunctionTask(func, args, kwargs, **task_kwargs)
        return self.submit_task(task)
    
    def submit_async_function(
        self,
        func: Callable[..., Awaitable[Any]],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        **task_kwargs
    ) -> str:
        """Submit an async function as a task."""
        task = AsyncFunctionTask(func, args, kwargs, **task_kwargs)
        return self.submit_task(task)
    
    async def wait_for_task(self, task_id: str, timeout: Optional[timedelta] = None) -> TaskResult:
        """Wait for a specific task to complete."""
        start_time = time.time()
        
        while True:
            if task_id in self._completed_tasks:
                return self._completed_tasks[task_id]
            
            if timeout and (time.time() - start_time) > timeout.total_seconds():
                raise TaskTimeoutError(f"Task {task_id} timed out after {timeout}")
            
            await asyncio.sleep(0.1)
    
    async def wait_for_all_tasks(self, timeout: Optional[timedelta] = None) -> List[TaskResult]:
        """Wait for all tasks to complete."""
        start_time = time.time()
        
        while True:
            if len(self._running_tasks) == 0 and self._task_queue.empty():
                return list(self._completed_tasks.values())
            
            if timeout and (time.time() - start_time) > timeout.total_seconds():
                raise TaskTimeoutError(f"Not all tasks completed within {timeout}")
            
            await asyncio.sleep(0.1)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a specific task."""
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id].status
        elif task_id in self._running_tasks:
            return TaskStatus.RUNNING
        elif task_id in self._tasks:
            return self._tasks[task_id].status
        return None
    
    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get the progress of a specific task."""
        if task_id in self._tasks:
            return self._tasks[task_id].progress
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        with self._lock:
            if task_id in self._running_tasks:
                self._running_tasks[task_id].cancel()
                del self._running_tasks[task_id]
                self._stats['cancelled_tasks'] += 1
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            return {
                **self._stats,
                'running_tasks': len(self._running_tasks),
                'queued_tasks': self._task_queue.qsize(),
                'completed_tasks': len(self._completed_tasks)
            }
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Check if we can start new tasks
                if len(self._running_tasks) < self.max_workers and not self._task_queue.empty():
                    try:
                        _, _, task = self._task_queue.get_nowait()
                        asyncio_task = asyncio.create_task(self._execute_task(task))
                        self._running_tasks[task.task_id] = asyncio_task
                    except queue.Empty:
                        pass
                
                # Clean up completed tasks
                completed_task_ids = []
                for task_id, asyncio_task in self._running_tasks.items():
                    if asyncio_task.done():
                        completed_task_ids.append(task_id)
                
                for task_id in completed_task_ids:
                    asyncio_task = self._running_tasks.pop(task_id)
                    try:
                        result = await asyncio_task
                        self._completed_tasks[task_id] = result
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Task {task_id} failed: {e}")
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            if self.logger:
                self.logger.debug(f"Starting task: {task.task_id}")
            
            # Execute with timeout if specified
            if task.timeout:
                result = await asyncio.wait_for(
                    task.execute(),
                    timeout=task.timeout.total_seconds()
                )
            else:
                result = await task.execute()
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                start_time=task.started_at,
                end_time=task.completed_at,
                metadata=task.metadata
            )
            
            self._stats['completed_tasks'] += 1
            if task_result.duration:
                self._stats['total_execution_time'] += task_result.duration
            
            if self.logger:
                self.logger.debug(f"Completed task: {task.task_id}")
            
            return task_result
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.completed_at = datetime.now()
            
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.TIMEOUT,
                error=TaskTimeoutError(f"Task {task.task_id} timed out"),
                start_time=task.started_at,
                end_time=task.completed_at,
                metadata=task.metadata
            )
            
            self._stats['failed_tasks'] += 1
            
            if self.logger:
                self.logger.warning(f"Task timed out: {task.task_id}")
            
            return task_result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=e,
                start_time=task.started_at,
                end_time=task.completed_at,
                metadata=task.metadata
            )
            
            self._stats['failed_tasks'] += 1
            
            if self.logger:
                self.logger.error(f"Task failed: {task.task_id}, error: {e}")
            
            return task_result


# Global task scheduler instance
_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get the global task scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


async def start_scheduler() -> None:
    """Start the global task scheduler."""
    scheduler = get_scheduler()
    await scheduler.start()


async def stop_scheduler() -> None:
    """Stop the global task scheduler."""
    global _scheduler
    if _scheduler:
        await _scheduler.stop()
        _scheduler = None


def submit_task(task: Task) -> str:
    """Submit a task to the global scheduler."""
    return get_scheduler().submit_task(task)


def submit_function(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    **task_kwargs
) -> str:
    """Submit a function to the global scheduler."""
    return get_scheduler().submit_function(func, args, kwargs, **task_kwargs)


def submit_async_function(
    func: Callable[..., Awaitable[Any]],
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    **task_kwargs
) -> str:
    """Submit an async function to the global scheduler."""
    return get_scheduler().submit_async_function(func, args, kwargs, **task_kwargs)
