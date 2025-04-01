# tasks.py
from typing import Dict, Any, List, Callable, Optional, Sequence
from dataclasses import dataclass, field
from eva import (
    Instance, VerificationResult, VerifiedTask, Agent, run, 
    MarketInstance, log, LogLevel
)


class TradingVerifiedTask(VerifiedTask[Dict[str, Any], str, bool, Dict[str, Any]]):
    """A task for a trading goal verified by checking portfolio state."""
    
    @staticmethod
    def create(
        instruction: str,
        snapshot_id: str,
        verification_function: Callable[[Dict[str, Any]], bool],
        verification_message: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> 'TradingVerifiedTask':
        """
        Create a trading verified task.
        
        Args:
            instruction: The goal to accomplish in the market
            snapshot_id: The snapshot ID to start from
            verification_function: Function that determines if the goal was achieved
            verification_message: Message explaining what constitutes success
            metadata: Optional metadata for the task
            
        Returns:
            A TradingVerifiedTask instance
        """
        log(LogLevel.INFO, f"Creating trading task: {instruction}", 
            extra={"snapshot_id": snapshot_id, "verification_message": verification_message})
        
        def trading_verifier(state: Instance[Dict[str, Any], Dict[str, Any]], 
                      actions: Sequence[str]) -> VerificationResult[bool]:
            log(LogLevel.INFO, f"Verifying trading task", 
                extra={"task": instruction, "action_count": len(actions)})
            
            # Extract portfolio state from the Instance
            portfolio_state = state.state.get("portfolio", {})
            market_state = state.state.get("market_data", {})
            log(LogLevel.INFO, f"Portfolio summary", 
                extra={"portfolio": portfolio_state, "action_count": len(actions)})
            
            # Check if the goal is achieved using the verification function
            try:
                combined_state = {
                    "portfolio": portfolio_state,
                    "market_data": market_state,
                    "actions": actions
                }
                success = verification_function(combined_state)
                
                if success:
                    log(LogLevel.SUCCESS, f"Goal achieved", 
                        extra={"task": instruction, "actions_taken": len(actions)})
                    return VerificationResult(
                        value=True,
                        success=True,
                        message=f"Goal achieved: {instruction}",
                        details={
                            "actions_taken": len(actions)
                        }
                    )
                else:
                    log(LogLevel.INFO, f"Goal not yet achieved", 
                        extra={"task": instruction, "verification_message": verification_message})
                    return VerificationResult(
                        value=False,
                        success=False,
                        message=f"Goal not yet achieved: {instruction}",
                        details={
                            "verification_message": verification_message
                        }
                    )
            except Exception as e:
                log(LogLevel.ERROR, f"Error in verification", 
                    extra={"error": str(e), "task": instruction})
                return VerificationResult(
                    value=False,
                    success=False,
                    message=f"Verification error: {str(e)}",
                    details={"error": str(e)}
                )
            
        return TradingVerifiedTask(
            instruction=instruction,
            snapshot_id=snapshot_id,
            verifier=trading_verifier,
            metadata=metadata or {}
        )


# ============= Verification Functions =============

def verify_portfolio_rebalanced(state: Dict[str, Any]) -> bool:
    """Verify portfolio has been rebalanced to target allocation."""
    portfolio = state.get("portfolio", {})
    target_allocation = portfolio.get("target_allocation", {})
    current_allocation = portfolio.get("current_allocation", {})
    
    # Check if current allocation is within 2% of target for each asset
    if not target_allocation or not current_allocation:
        return False
    
    tolerance = 0.02  # 2% tolerance
    
    for asset, target_pct in target_allocation.items():
        current_pct = current_allocation.get(asset, 0)
        if abs(current_pct - target_pct) > tolerance:
            return False
    
    return True

def verify_risk_reduction(state: Dict[str, Any]) -> bool:
    """Verify portfolio risk has been reduced."""
    portfolio = state.get("portfolio", {})
    initial_risk = portfolio.get("initial_risk_metrics", {}).get("portfolio_volatility", 0)
    current_risk = portfolio.get("risk_metrics", {}).get("portfolio_volatility", 0)
    
    # Verify risk has been reduced by at least 10%
    if initial_risk == 0:
        return False
    
    risk_reduction = (initial_risk - current_risk) / initial_risk
    return risk_reduction >= 0.10  # 10% reduction

def verify_profit_target(state: Dict[str, Any]) -> bool:
    """Verify if profit target has been reached."""
    portfolio = state.get("portfolio", {})
    initial_value = portfolio.get("initial_value", 0)
    current_value = portfolio.get("current_value", 0)
    target_return = portfolio.get("target_return", 0.05)  # Default 5%
    
    if initial_value == 0:
        return False
    
    actual_return = (current_value - initial_value) / initial_value
    return actual_return >= target_return

def verify_dollar_cost_average(state: Dict[str, Any]) -> bool:
    """Verify if dollar cost averaging has been implemented."""
    actions = state.get("actions", [])
    
    # Parse actions to find purchase patterns
    purchase_timestamps = {}
    
    for action in actions:
        if "buy:" in action:  # Example format: "buy:AAPL:100"
            parts = action.split(":")
            if len(parts) >= 3:
                ticker = parts[1]
                if ticker not in purchase_timestamps:
                    purchase_timestamps[ticker] = []
                purchase_timestamps[ticker].append(parts[3] if len(parts) > 3 else None)
    
    # Need at least one asset with 3+ purchases
    for ticker, timestamps in purchase_timestamps.items():
        if len(timestamps) >= 3:
            return True
    
    return False

def verify_sector_diversification(state: Dict[str, Any]) -> bool:
    """Verify if portfolio is diversified across sectors."""
    portfolio = state.get("portfolio", {})
    sector_exposure = portfolio.get("sector_exposure", {})
    
    # Need at least 4 sectors with no more than 30% in any one sector
    if len(sector_exposure) < 4:
        return False
    
    for sector, weight in sector_exposure.items():
        if weight > 0.30:  # No more than 30% in any sector
            return False
    
    return True

# Store verification functions in a registry for lookup by name
VERIFICATION_FUNCTIONS = {
    "verify_portfolio_rebalanced": verify_portfolio_rebalanced,
    "verify_risk_reduction": verify_risk_reduction,
    "verify_profit_target": verify_profit_target,
    "verify_dollar_cost_average": verify_dollar_cost_average,
    "verify_sector_diversification": verify_sector_diversification,
}

# ============= Task Definition =============

@dataclass
class TaskDefinition:
    """Definition of a trading task with verification details."""
    id: str  # Unique identifier for the task
    instruction: str  # Human-readable instruction
    verification_fn_name: str  # Name of verification function
    verification_message: str  # Message to show if verification fails
    snapshot_id: str = ""  # Starting snapshot ID (can be set at runtime)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_verification_function(self) -> Callable[[Dict[str, Any]], bool]:
        """Get the verification function for this task."""
        if self.verification_fn_name not in VERIFICATION_FUNCTIONS:
            raise ValueError(f"Unknown verification function: {self.verification_fn_name}")
        return VERIFICATION_FUNCTIONS[self.verification_fn_name]

# ============= Task Registry =============

# Define all available tasks
REGISTERED_TASKS = [
    TaskDefinition(
        id="rebalance-portfolio",
        instruction="Rebalance the portfolio to match target allocation",
        verification_fn_name="verify_portfolio_rebalanced",
        verification_message="Adjust holdings to match target allocation within 2% tolerance.",
        metadata={"category": "Portfolio Management", "difficulty": "Medium"},
    ),
    TaskDefinition(
        id="reduce-risk",
        instruction="Reduce portfolio volatility by at least 10%",
        verification_fn_name="verify_risk_reduction",
        verification_message="Reduce portfolio volatility by at least 10% through position adjustments.",
        metadata={"category": "Risk Management", "difficulty": "Hard"},
    ),
    TaskDefinition(
        id="reach-profit-target",
        instruction="Reach 5% portfolio return target",
        verification_fn_name="verify_profit_target",
        verification_message="Execute trades to achieve 5% portfolio return.",
        metadata={"category": "Performance", "difficulty": "Hard"},
    ),
    TaskDefinition(
        id="implement-dca",
        instruction="Implement dollar cost averaging for selected assets",
        verification_fn_name="verify_dollar_cost_average",
        verification_message="Execute regular purchases of the same assets over time.",
        metadata={"category": "Strategy", "difficulty": "Easy"},
    ),
    TaskDefinition(
        id="sector-diversification",
        instruction="Diversify the portfolio across at least 4 sectors",
        verification_fn_name="verify_sector_diversification",
        verification_message="Ensure portfolio has exposure to at least 4 sectors with no more than 30% in any one sector.",
        metadata={"category": "Portfolio Management", "difficulty": "Medium"},
    ),
]

# ============= Helper Functions =============

def get_task_by_id(task_id: str) -> Optional[TaskDefinition]:
    """Get a task definition by its ID."""
    for task in REGISTERED_TASKS:
        if task.id == task_id:
            return task
    return None

def create_trading_verified_task(task_id: str, snapshot_id: str = None):
    """
    Create a TradingVerifiedTask instance for the given task ID.
    
    Example usage:
    ```
    from tasks import create_trading_verified_task
    
    # Create task
    task = create_trading_verified_task("rebalance-portfolio", "snap_12345")
    
    # Run task with agent
    from eva import run
    result, _ = await run(task=task, agent=agent, max_steps=100)
    ```
    """
    task_def = get_task_by_id(task_id)
    if not task_def:
        raise ValueError(f"Unknown task ID: {task_id}")
    
    # Set snapshot ID if provided
    if snapshot_id:
        task_def.snapshot_id = snapshot_id
    
    return TradingVerifiedTask.create(
        instruction=task_def.instruction,
        snapshot_id=task_def.snapshot_id,
        verification_function=task_def.get_verification_function(),
        verification_message=task_def.verification_message,
        metadata=task_def.metadata
    )