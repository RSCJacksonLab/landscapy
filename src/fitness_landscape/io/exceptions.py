class BundleIOError(Exception):
    """Base error for portable landscape bundle I/O."""


class BundleValidationError(BundleIOError):
    """Raised when a bundle manifest or payload is invalid."""


class ChecksumMismatchError(BundleValidationError):
    """Raised when a bundle payload checksum does not match the manifest."""
