# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `SolidBodyContact`.
- Added the arguments `facing` and `self_contact` to `SolidBodyContact`. A face-pair is
  only considered by the contact search if the outward unit normal vectors of both faces
  are opposed and if both faces belong to different bodies of the mesh. This makes it
  possible to use all faces on the outline of a mesh as contact surfaces, i.e. without
  restricting the boundary regions to the region of interest.

### Changed
- Changed the broad-phase contact search to query the faces of each body of the primary
  surface separately. Otherwise the faces of the own body of an integration point, which
  are always the closest ones, occupy the candidates of a query and hide the faces of
  the other bodies.
