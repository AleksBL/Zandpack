"""Backward-compatibility alias (2026-10-10 rename).

The active implementation was renamed AI_fit.py -> AI_fit_base.py.
This file makes the module name `AI_fit` an ALIAS of AI_fit_base
(same module object), so every script that does `import AI_fit` —
including attribute access to internal helpers not in __all__
(lorentzian_basis, _poles_init_dispatch, psd_refine, ...) — keeps
working, and pickled fit caches (whose objects reference the module
NAME 'AI_fit') keep unpickling.

New code should import AI_fit_base directly. Remove this alias only
after regenerating every cache and updating every script.
"""
import sys as _sys

import AI_fit_base as _base

_sys.modules[__name__] = _base
