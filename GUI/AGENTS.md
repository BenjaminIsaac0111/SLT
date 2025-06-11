# GUI Contribution Guidelines

This directory houses the PyQt5 interface following a simplified Model/View/Controller pattern.

## Directory layout
- `controllers/` – mediators between views and models.
- `views/` – Qt widgets and dialogs.
- `models/` – domain logic used by the GUI.
- `workers/` – background tasks for long running operations.
- `assets/` – icons and other resources.
- `unittests/` – unit tests for GUI components.

## Coding conventions
- Target **Python 3.8** and follow PEP8 with 4 spaces per indent.
- Use Qt's signal/slot mechanism when communicating across threads.
- Delegate blocking work to `workers/` using `QThreadPool` to keep the UI responsive.
- Provide docstrings for all public classes, functions and modules.
- Add unit tests under `unittests/` for new features.
