"""Per-item query budget : the counter the Phase-3 localhost victim wrapper wraps.
Counts attack-phase queries only; a charge that would exceed the limit is refused BEFORE
any inference, so executed inferences == counted queries (ARCH section 6.3)."""
from __future__ import annotations


class BudgetExhausted(Exception):
    """Raised by QueryBudget.charge when the next query would exceed the limit."""


class QueryBudget:
    def __init__(self, limit: int) -> None:
        if int(limit) < 1:
            raise ValueError("query budget must be at least 1")
        self.limit = int(limit)
        self.used = 0

    def charge(self, n: int = 1) -> int:
        """Charge n queries; raise BudgetExhausted (nothing charged) if they would exceed the limit."""
        if int(n) < 1:
            raise ValueError("charge needs n >= 1")
        if self.used + int(n) > self.limit:
            raise BudgetExhausted(f"budget {self.limit} exhausted: {self.used} used, {n} requested")
        self.used += int(n)
        return self.used

    @property
    def remaining(self) -> int:
        return self.limit - self.used
