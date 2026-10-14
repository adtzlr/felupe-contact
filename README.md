<p align="center">
  <p align="center"><b>felupe-contact</b></p>
  <p align="center"><i>Contact Extension-package for FElupe.</i></p>
</p>

[![FElupe](https://img.shields.io/badge/%F0%9F%94%8D-FElupe-white)](https://felupe.readthedocs.io/en/stable/) [![Documentation Status](https://readthedocs.org/projects/felupe-contact/badge/?version=latest)](https://felupe-contact.readthedocs.io/en/latest/?badge=latest) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip) [![codecov](https://codecov.io/gh/adtzlr/felupe-contact/branch/main/graph/badge.svg)](https://codecov.io/gh/adtzlr/felupe-contact) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

felupe-contact is an extension package for [FElupe](https://github.com/adtzlr/felupe) with contact formulations between the surfaces of (hyperelastic) solid bodies, which undergo large deformations and large relative displacements.

> [!WARNING]
> felupe-contact is in an early stage of development. It is not available on PyPI yet and its API may change without a deprecation.

## ✨ Highlights
- ✅ a [contact between the surfaces of solid bodies](https://felupe-contact.readthedocs.io/en/latest/felupe_contact.html#felupe_contact.SolidBodyContact) (surface-to-surface, penalty-based) with optional [Coulomb friction](https://felupe-contact.readthedocs.io/en/latest/howto/friction.html)
- ✅ three-dimensional, plane strain and axisymmetric problems
- ✅ all boundary regions of FElupe, i.e. faces with linear or quadratic shape functions
- ✅ an item of FElupe, i.e. it is used in a `Step` and in a `Job` like a `SolidBody`
- ✅ created directly on solid bodies with all faces on the outline of their meshes, i.e. no boundary regions are required
- ✅ an efficient and robust contact search for contact surfaces with all faces, e.g. of many bodies

## 📦 Installation
Install Python, open a terminal and run

```shell
pip install git+https://github.com/adtzlr/felupe-contact.git@main
```

The [documentation](https://felupe-contact.readthedocs.io/en/latest/) covers more details, like the required and optional dependencies.

## 🚀 Getting Started
This minimal code-block demonstrates a rubber block, which is pressed on a (stiffer) block. Both blocks are meshed individually and are combined to a single mesh.

```python
import felupe as fem

from felupe_contact import SolidBodyContact

bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(3, 3, 3))
mesh = fem.MeshContainer([bottom, top], merge=True).stack()

region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])
solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

# the contact surfaces are created on the outline of the mesh of the solid body
contact = SolidBodyContact(solid)

boundaries = {
    "fixed": fem.Boundary(field[0], fz=0.0),
    "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
    "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
}
step = fem.Step(
    items=[solid, contact],
    ramp={boundaries["move"]: fem.math.linsteps([0, -0.3], num=3)},
    boundaries=boundaries,
)
job = fem.Job(steps=[step]).evaluate()

solid.plot("Principal Values of Cauchy Stress").show()
```

## 📖 Documentation
The documentation is located [here](https://felupe-contact.readthedocs.io/en/latest/). It contains a [Getting Started](https://felupe-contact.readthedocs.io/en/latest/start.html) guide, which adds a contact to a FElupe-script step by step, How-To guides, a gallery of examples and the API reference.

## 🛠️ Testing

To run the unit tests, check out this repository and type

```
tox
```

The tests are run with the latest release of FElupe. The development version of FElupe (its `main` branch) is tested by

```
tox -e main
```

## 📄 Changelog
All notable changes to this project will be documented in [this file](CHANGELOG.md). The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 🔓 License
felupe-contact - Contact Extension-package for FElupe (C) 2026 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
