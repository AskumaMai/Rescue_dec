"""rescue package

Lightweight package initializer so the package can be imported as
``import rescue``. It exposes a version and defines the public
subpackages. Keep this file minimal to avoid import-time side-effects.
"""

__all__ = [
	"simulate",
	"model",
	"create_dataset",
	"rescue",
]

__version__ = "0.1.0"

# Do not import heavy submodules at package import time. Users may
# import submodules explicitly: `from rescue import simulate`.
