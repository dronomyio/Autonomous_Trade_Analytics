"""
E.V.A. Framework: Executions with Verified Agents for Trading
Adapted from the Pokemon example for stock trading applications.
"""

import asyncio
import inspect
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union
import json

# Type variables for generic components
S = TypeVar('S')  # State type
A = TypeVar('A')  # Action type
R = TypeVar('R')  # Verification result type
T = TypeVar('T')  # Snapshot type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trade_analytics")

# Define log levels
class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"

def log(level: str, message: str, extra: Optional[Dict[str, Any]] = None):
    """Log a message with structured data."""
    extra = extra or {}
    
    if level == LogLevel.DEBUG:
        logger.debug(message, extra=extra)
    elif level == LogLevel.INFO:
        logger.info(message, extra=extra)
    elif level == LogLevel.WARNING:
        logger.warning(message, extra=extra)
    elif level == LogLevel.ERROR:
        logger.error(message, extra=extra)
    elif level == LogLevel.SUCCESS:
        logger.info(f"✅ {message}", extra=extra)
    else:
        logger.info(message, extra=extra)
    
    # Also print to console for convenience
    level_prefix = {
        LogLevel.DEBUG: "🔍",
        LogLevel.INFO: "ℹ️",
        LogLevel.WARNING: "⚠️",
        LogLevel.ERROR: "❌",
        LogLevel.SUCCESS: "✅"
    }.get(level, "ℹ️")
    
    print(f"{level_prefix} {message}")
    if extra and len(extra) > 0:
        print(f"  {json.dumps(extra, default=str)}")

@dataclass
class VerificationResult(Generic[R]):
    """Result of verifying an agent's progress on a task."""
    value: R  # The result value
    success: bool  # Whether the verification was successful
    message: str  # A message explaining the result
    details: Dict[str, Any] = field(default_factory=dict)  # Additional details

@dataclass
class TrajectoryStep(Generic[S, A, R, T]):
    """A step in an agent's trajectory, recording state, action, and result."""
    state: "Instance[S, T]"  # The state at this step
    snapshot: T  # A snapshot of the state for visualization/rollback
    action: Optional[A] = None  # The action taken (None for initial state)
    result: Optional[VerificationResult[R]] = None  # Result of verification
    timestamp: datetime = field(default_factory=datetime.now)  # When step occurred

class Trajectory(Generic[S, A, R, T]):
    """Records the sequence of steps taken by an agent."""
    
    def __init__(self):
        """Initialize an empty trajectory."""
        self.steps: List[TrajectoryStep[S, A, R, T]] = []
    
    def add_step(self, state: "Instance[S, T]", action: Optional[A] = None,
                 result: Optional[VerificationResult[R]] = None) -> None:
        """Add a step to the trajectory."""
        step = TrajectoryStep(
            state=state,
            snapshot=state.snapshot(),
            action=action,
            result=result,
            timestamp=datetime.now()
        )
        self.steps.append(step)
        log(LogLevel.DEBUG, f"Added trajectory step {len(self.steps)-1}")
    
    def get_last_step(self) -> Optional[TrajectoryStep[S, A, R, T]]:
        """Get the last step in the trajectory."""
        if not self.steps:
            return None
        return self.steps[-1]
    
    def get_step(self, index: int) -> Optional[TrajectoryStep[S, A, R, T]]:
        """Get a step by index."""
        if index < 0 or index >= len(self.steps):
            return None
        return self.steps[index]

class Instance(Generic[S, T]):
    """A wrapper around a state that can create snapshots."""
    
    def __init__(self, state: S):
        """Initialize with a state."""
        self.state = state
    
    def snapshot(self) -> T:
        """Create a snapshot for visualization and rollback."""
        # Override this in subclasses to implement actual snapshot creation
        return None  # type: ignore

class VerifiedTask(Generic[S, A, R, T]):
    """A task with a verification function to check if it's been accomplished."""
    
    def __init__(
        self,
        instruction: str,
        snapshot_id: str,
        verifier: Callable[["Instance[S, T]", Sequence[A]], VerificationResult[R]],
        metadata: Dict[str, str] = None
    ):
        """Initialize a verified task."""
        self.instruction = instruction
        self.snapshot_id = snapshot_id
        self.verifier = verifier
        self.metadata = metadata or {}
    
    def verify(self, state: "Instance[S, T]", actions: Sequence[A]) -> VerificationResult[R]:
        """Verify if the task has been accomplished."""
        return self.verifier(state, actions)

class Agent(Generic[S, A, R, T]):
    """An agent that can take actions based on state."""
    
    def __init__(self):
        """Initialize an agent."""
        self.trajectory: Trajectory[S, A, R, T] = Trajectory()
    
    async def initialize_state(self, **kwargs) -> Instance[S, T]:
        """Initialize the state."""
        raise NotImplementedError
    
    async def run_step(self, state: Instance[S, T]) -> A:
        """Determine the next action to take."""
        raise NotImplementedError
    
    async def apply_action(self, state: S, action: A) -> S:
        """Apply an action to a state."""
        raise NotImplementedError

class MarketInstance(Instance[Dict[str, Any], str]):
    """A specialized instance for market data and portfolio state."""
    
    def __init__(self, state: Dict[str, Any], connector=None):
        """Initialize a market instance."""
        super().__init__(state)
        self._connector = connector
    
    def snapshot(self) -> str:
        """Create a snapshot ID with timestamp."""
        # In a real application, you might save state to disk or database
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"market_snapshot_{timestamp}"

async def run(
    task: VerifiedTask[S, A, R, T],
    agent: Agent[S, A, R, T],
    max_steps: int = 100,
    verify_every_step: bool = True,
    ttl_seconds: int = 3600
) -> Tuple[VerificationResult[R], Trajectory[S, A, R, T]]:
    """Run an agent on a task until it's verified or max_steps is reached."""
    
    # Initialize state
    log(LogLevel.INFO, f"Initializing state for task: {task.instruction}")
    state = await agent.initialize_state(snapshot_id=task.snapshot_id)
    
    # Add initial state to trajectory
    agent.trajectory.add_step(state)
    
    # Track actions taken
    actions: List[A] = []
    
    # Set timeout based on TTL
    start_time = time.time()
    
    # Run steps until max_steps or timeout
    step = 0
    while step < max_steps:
        # Check if we've exceeded TTL
        elapsed = time.time() - start_time
        if elapsed > ttl_seconds:
            log(LogLevel.WARNING, f"TTL exceeded ({elapsed:.1f}s > {ttl_seconds}s)")
            result = VerificationResult(
                value=None,  # type: ignore
                success=False,
                message="Timeout: TTL exceeded",
                details={"elapsed_seconds": elapsed, "ttl_seconds": ttl_seconds}
            )
            return result, agent.trajectory
        
        log(LogLevel.INFO, f"Starting step {step}/{max_steps} (elapsed: {elapsed:.1f}s)")
        
        # Get action from agent
        try:
            action = await agent.run_step(state)
        except Exception as e:
            log(LogLevel.ERROR, f"Error in agent.run_step: {e}")
            break
        
        log(LogLevel.INFO, f"Agent action: {action}")
        actions.append(action)
        
        # Apply action to get new state
        try:
            new_state_dict = await agent.apply_action(state.state, action)
            state = MarketInstance(new_state_dict, getattr(state, '_connector', None))
        except Exception as e:
            log(LogLevel.ERROR, f"Error in agent.apply_action: {e}")
            break
        
        # Verify if task is accomplished
        if verify_every_step:
            result = task.verify(state, actions)
            
            # Add step with verification result to trajectory
            agent.trajectory.add_step(state, action, result)
            
            if result.success:
                log(LogLevel.SUCCESS, f"Task verified successfully: {result.message}")
                return result, agent.trajectory
        else:
            # Add step without verification to trajectory
            agent.trajectory.add_step(state, action)
        
        step += 1
    
    # Final verification if we didn't verify every step
    if not verify_every_step:
        result = task.verify(state, actions)
        if result.success:
            log(LogLevel.SUCCESS, f"Task verified successfully: {result.message}")
            return result, agent.trajectory
    
    # If we get here, we didn't succeed within max_steps
    log(LogLevel.WARNING, f"Task not verified within {max_steps} steps")
    result = VerificationResult(
        value=None,  # type: ignore
        success=False,
        message=f"Task not accomplished within {max_steps} steps",
        details={"max_steps": max_steps, "steps_taken": step}
    )
    return result, agent.trajectory