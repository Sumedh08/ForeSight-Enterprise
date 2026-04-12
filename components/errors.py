class ComponentBlockedError(RuntimeError):
    """Raised when a component cannot run safely with the current configuration."""


class ModelAccessError(ComponentBlockedError):
    """Raised when a configured model backend is unavailable."""


class BenchmarkAccessError(ModelAccessError):
    """Raised when benchmark mode is requested without the required NIM credentials."""
