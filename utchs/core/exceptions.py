"""Custom exceptions for the UTCHS framework."""

from typing import Any, Dict, Optional


class UTCHSError(Exception):
    """Base exception class for all UTCHS framework errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(UTCHSError):
    """Exception raised for validation errors."""

    pass


class ConfigurationError(UTCHSError):
    """Exception raised for configuration errors."""

    pass


class ResourceError(UTCHSError):
    """Exception raised for resource-related errors."""

    pass


class ModelError(UTCHSError):
    """Exception raised for model-related errors."""

    pass


class DataError(UTCHSError):
    """Exception raised for data-related errors."""

    pass


class ComputationError(UTCHSError):
    """Exception raised for computation-related errors."""

    pass


class IOError(UTCHSError):
    """Exception raised for I/O-related errors."""

    pass


class StateError(UTCHSError):
    """Exception raised for state-related errors."""

    pass


class DependencyError(UTCHSError):
    """Exception raised for dependency-related errors."""

    pass


class SecurityError(UTCHSError):
    """Exception raised for security-related errors."""

    pass


class TimeoutError(UTCHSError):
    """Exception raised for timeout errors."""

    pass


class NotImplementedError(UTCHSError):
    """Exception raised for not implemented features."""

    pass


class DeprecationError(UTCHSError):
    """Exception raised for deprecated features."""

    pass


class VersionError(UTCHSError):
    """Exception raised for version-related errors."""

    pass


class RegistryError(UTCHSError):
    """Exception raised for registry-related errors."""

    pass


class LoggingError(UTCHSError):
    """Exception raised for logging-related errors."""

    pass


class MetricsError(UTCHSError):
    """Exception raised for metrics-related errors."""

    pass


class VisualizationError(UTCHSError):
    """Exception raised for visualization-related errors."""

    pass


class OptimizationError(UTCHSError):
    """Exception raised for optimization-related errors."""

    pass


class ParallelizationError(UTCHSError):
    """Exception raised for parallelization-related errors."""

    pass


class CachingError(UTCHSError):
    """Exception raised for caching-related errors."""

    pass


class SerializationError(UTCHSError):
    """Exception raised for serialization-related errors."""

    pass


class DeserializationError(UTCHSError):
    """Exception raised for deserialization-related errors."""

    pass


class CompressionError(UTCHSError):
    """Exception raised for compression-related errors."""

    pass


class EncryptionError(UTCHSError):
    """Exception raised for encryption-related errors."""

    pass


class AuthenticationError(UTCHSError):
    """Exception raised for authentication-related errors."""

    pass


class AuthorizationError(UTCHSError):
    """Exception raised for authorization-related errors."""

    pass


class RateLimitError(UTCHSError):
    """Exception raised for rate limit errors."""

    pass


class QuotaError(UTCHSError):
    """Exception raised for quota-related errors."""

    pass


class ResourceExhaustedError(UTCHSError):
    """Exception raised when resources are exhausted."""

    pass


class ResourceUnavailableError(UTCHSError):
    """Exception raised when resources are unavailable."""

    pass


class ResourceConflictError(UTCHSError):
    """Exception raised when there is a resource conflict."""

    pass


class ResourceNotFoundError(UTCHSError):
    """Exception raised when a resource is not found."""

    pass


class ResourceAlreadyExistsError(UTCHSError):
    """Exception raised when a resource already exists."""

    pass


class ResourceInvalidError(UTCHSError):
    """Exception raised when a resource is invalid."""

    pass


class ResourceExpiredError(UTCHSError):
    """Exception raised when a resource has expired."""

    pass


class ResourceLockedError(UTCHSError):
    """Exception raised when a resource is locked."""

    pass


class ResourceBusyError(UTCHSError):
    """Exception raised when a resource is busy."""

    pass


class ResourceDeletedError(UTCHSError):
    """Exception raised when a resource has been deleted."""

    pass


class ResourceModifiedError(UTCHSError):
    """Exception raised when a resource has been modified."""

    pass


class ResourceVersionMismatchError(UTCHSError):
    """Exception raised when there is a resource version mismatch."""

    pass


class ResourceStateError(UTCHSError):
    """Exception raised when there is a resource state error."""

    pass


class ResourcePermissionError(UTCHSError):
    """Exception raised when there is a resource permission error."""

    pass


class ResourceQuotaError(UTCHSError):
    """Exception raised when there is a resource quota error."""

    pass


class ResourceRateLimitError(UTCHSError):
    """Exception raised when there is a resource rate limit error."""

    pass


class ResourceTimeoutError(UTCHSError):
    """Exception raised when there is a resource timeout error."""

    pass


class ResourceConnectionError(UTCHSError):
    """Exception raised when there is a resource connection error."""

    pass


class ResourceNetworkError(UTCHSError):
    """Exception raised when there is a resource network error."""

    pass


class ResourceProtocolError(UTCHSError):
    """Exception raised when there is a resource protocol error."""

    pass


class ResourceFormatError(UTCHSError):
    """Exception raised when there is a resource format error."""

    pass


class ResourceEncodingError(UTCHSError):
    """Exception raised when there is a resource encoding error."""

    pass


class ResourceDecodingError(UTCHSError):
    """Exception raised when there is a resource decoding error."""

    pass


class ResourceParsingError(UTCHSError):
    """Exception raised when there is a resource parsing error."""

    pass


class ResourceValidationError(UTCHSError):
    """Exception raised when there is a resource validation error."""

    pass


class ResourceTransformationError(UTCHSError):
    """Exception raised when there is a resource transformation error."""

    pass


class ResourceProcessingError(UTCHSError):
    """Exception raised when there is a resource processing error."""

    pass


class ResourceExecutionError(UTCHSError):
    """Exception raised when there is a resource execution error."""

    pass


class ResourceInitializationError(UTCHSError):
    """Exception raised when there is a resource initialization error."""

    pass


class ResourceCleanupError(UTCHSError):
    """Exception raised when there is a resource cleanup error."""

    pass


class ResourceRecoveryError(UTCHSError):
    """Exception raised when there is a resource recovery error."""

    pass


class ResourceBackupError(UTCHSError):
    """Exception raised when there is a resource backup error."""

    pass


class ResourceRestoreError(UTCHSError):
    """Exception raised when there is a resource restore error."""

    pass


class ResourceMigrationError(UTCHSError):
    """Exception raised when there is a resource migration error."""

    pass


class ResourceUpgradeError(UTCHSError):
    """Exception raised when there is a resource upgrade error."""

    pass


class ResourceDowngradeError(UTCHSError):
    """Exception raised when there is a resource downgrade error."""

    pass


class ResourceCompatibilityError(UTCHSError):
    """Exception raised when there is a resource compatibility error."""

    pass


class ResourceDependencyError(UTCHSError):
    """Exception raised when there is a resource dependency error."""

    pass


class ResourceConfigurationError(UTCHSError):
    """Exception raised when there is a resource configuration error."""

    pass 