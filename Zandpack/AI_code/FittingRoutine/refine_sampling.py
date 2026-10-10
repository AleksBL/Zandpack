"""Backward-compatibility alias (2026-10-10 rename).

The historical refine_sampling.py shim was renamed ->
AI_fit_adaptivesampling.py. This file makes the module name
`refine_sampling` an ALIAS of AI_fit_adaptivesampling (which itself
aliases AI_fit_base's sampling/glue surface), so the ~90
driver/diagnostic scripts and the pickled fit caches (whose objects
reference the module NAME 'refine_sampling') keep working.

New code should import AI_fit_base (or AI_fit_adaptivesampling for
the sampling/glue surface) directly. Remove this alias only after
regenerating every cache and updating every script.
"""
import sys as _sys

import AI_fit_adaptivesampling as _base

_sys.modules[__name__] = _base
