# SPDX-License-Identifier: GPL-3.0-or-later

from .__about__ import __version__
from ._nitsche import SolidBodyContactNitsche
from ._solidbody_contact import SolidBodyContact

__all__ = [
    "__version__",
    "SolidBodyContact",
    "SolidBodyContactNitsche",
]
