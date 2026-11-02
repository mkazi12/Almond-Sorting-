"""Shared service validation and cooperative maintenance locking."""

import math

from sqlalchemy import text

from app.storage.errors import NotFoundError, ValidationError


MAINTENANCE_LOCK = 48490001


def writing(session):
    """Hold a shared lock until the caller commits; backups take it exclusively."""
    session.execute(text("SELECT pg_advisory_xact_lock_shared(:key)"),
                    {"key": MAINTENANCE_LOCK})


def require(session, model, identifier):
    row = session.get(model, identifier)
    if row is None:
        raise NotFoundError(f"{model.__name__} {identifier} was not found. Refresh the list and select it again.")
    return row


def nonblank(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be non-empty. Supply a value and try again.")
    return value.strip()


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{label} must be an integer of at least {minimum}.")
    return value


def finite(value, label, minimum=None, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValidationError(f"{label} must be finite and within {minimum}..{maximum}.")
    return value


def normalized_box(box, width, height):
    if not width or not height or width <= 0 or height <= 0:
        raise ValidationError("Media dimensions are unavailable. Import the recording again.")
    if len(box) != 4:
        raise ValidationError("A box must contain four pixel xyxy coordinates.")
    values = []
    for coordinate, scale in zip(box, (width, height, width, height)):
        value = finite(coordinate, "Box coordinate") / scale
        if value < -1e-6 or value > 1 + 1e-6:
            raise ValidationError("Box lies outside the image. Check model coordinates and image dimensions.")
        values.append(min(1., max(0., value)))
    x0, y0, x1, y1 = values
    if x0 >= x1 or y0 >= y1:
        raise ValidationError("Box must have positive width and height. Check xyxy coordinates.")
    return dict(zip(("x_min", "y_min", "x_max", "y_max"), values))
