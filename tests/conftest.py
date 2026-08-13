import pytest

from leb.just_focus import backend


@pytest.fixture(autouse=True)
def _reset_backend():
    """Reset the active backend/precision to the default after each test.

    Backend state is process-wide (see `leb.just_focus.backend`), so without
    this a test that calls `set_backend("torch")` would leak that choice into
    unrelated tests that run afterward.
    """
    yield
    backend.set_backend(backend.Backend.NUMPY, precision=backend.Precision.FLOAT64)
