class RecoverableToolError(Exception):
    """Tool error that should trigger retry, not loop break or circuit break."""
    pass

__INTERNAL__ = True


