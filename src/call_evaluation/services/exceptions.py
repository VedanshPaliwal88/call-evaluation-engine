"""Domain-specific exceptions for the call evaluation service layer."""


class LLMUnavailableError(RuntimeError):
    """Raised when an LLM-backed detector is requested without runtime support."""


class FileValidationError(ValueError):
    """Raised when an uploaded file cannot be parsed into a transcript."""
