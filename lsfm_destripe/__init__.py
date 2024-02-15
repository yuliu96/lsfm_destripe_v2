# -*- coding: utf-8 -*-

"""Top-level package for LSFM DeStripe PyTorch."""

__author__ = "Jianxu Chen"
__email__ = "jianxu.chen@isas.de"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.1"


def get_module_version():
    return __version__


from .core import DeStripe  # noqa: F401
