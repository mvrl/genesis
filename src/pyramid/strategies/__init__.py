"""Strategy registry. Import strategy modules so they self-register.

Add a new idea by creating ``strategies/<name>.py`` with an ``@register("<name>")``
class, then importing it here.
"""
from .base import GenContext, Strategy, get_strategy, list_strategies, register  # noqa: F401

# Register the supported strategy (import for side effects).
from . import unified           # noqa: F401,E402
