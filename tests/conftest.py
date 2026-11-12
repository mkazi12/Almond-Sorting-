"""Keep one offscreen QApplication alive for desktop tests."""

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config):
    config.addinivalue_line("markers", "gui: uses the shared offscreen QApplication")


def _uses_qt_application(item):
    return any(getattr(item.module, name, None) is not None
               for name in ("QApplication", "QCoreApplication", "QGuiApplication"))


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _uses_qt_application(item):
            item.add_marker(pytest.mark.gui)


@pytest.fixture(scope="session", autouse=True)
def qt_test_session(request):
    if any(_uses_qt_application(item) for item in request.session.items):
        request.getfixturevalue("qt_application")


@pytest.fixture(scope="session")
def qt_application():
    """Initialize QWidget support before any test can create QCoreApplication."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None and not isinstance(app, QApplication):
        pytest.fail("Qt tests need a QApplication; a QCoreApplication was created first.")
    app = app or QApplication([])
    yield app
