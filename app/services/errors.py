class ServiceError(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BadRequest(ServiceError):
    status_code = 422


class NotFound(ServiceError):
    status_code = 404


class UnsupportedMedia(ServiceError):
    status_code = 415


class PayloadTooLarge(ServiceError):
    status_code = 413


class TooManyRequests(ServiceError):
    status_code = 429


class ServiceUnavailable(ServiceError):
    status_code = 503


class PermanentRunError(Exception):
    """Raised inside the pipeline for errors that retrying cannot fix (corrupt audio...)."""
