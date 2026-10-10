# Zandpack/AI_code/FittingRoutine/__init__.py
import sys as _sys
from . import AI_fit_base as _base
_sys.modules.setdefault("AI_fit", _base)          # pickle compat
_sys.modules.setdefault("AI_fit_base", _base)     # flat-name compat
_sys.modules.setdefault("refine_sampling", _base) # historical name
