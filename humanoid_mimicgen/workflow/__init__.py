"""Independent structured manipulation workflow; no automatic execution on import."""
from .contracts import WorkflowError
from .store import Store
__all__ = ['WorkflowError', 'Store']
