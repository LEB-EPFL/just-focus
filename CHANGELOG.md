# Changelog

## [2.0.0](https://github.com/LEB-EPFL/just-focus/compare/v1.1.0...v2.0.0) (2026-08-14)


### ⚠ BREAKING CHANGES

* Pass gradients through Zernike polynomial coefficients ([#80](https://github.com/LEB-EPFL/just-focus/issues/80))

### Features

* Add PyTorch backend ([#78](https://github.com/LEB-EPFL/just-focus/issues/78)) ([5f4e8a9](https://github.com/LEB-EPFL/just-focus/commit/5f4e8a923a4020977c4a0aad5932813ed578ce80))
* Pass gradients through Zernike polynomial coefficients ([#80](https://github.com/LEB-EPFL/just-focus/issues/80)) ([38cf326](https://github.com/LEB-EPFL/just-focus/commit/38cf32648632ddc3572110ce6f18294bc7927529))


### Documentation

* Add CONTRIBUTING.md ([63e5725](https://github.com/LEB-EPFL/just-focus/commit/63e57258ecacfcbeac9efc2310752e03963072f9))

## [1.1.0](https://github.com/LEB-EPFL/just-focus/compare/v1.0.0...v1.1.0) (2026-08-11)


### Features

* Add a script demonstrating adding aberrations to a halfmoon beam ([bc43392](https://github.com/LEB-EPFL/just-focus/commit/bc43392ba1a10dfae635fd80121dc9d19fb67b17))
* Zernike polynomial-based pupil phase perturbations ([#77](https://github.com/LEB-EPFL/just-focus/issues/77)) ([5f231a8](https://github.com/LEB-EPFL/just-focus/commit/5f231a8883c857df19afb6483361f70ca920f431))


### Documentation

* Remove import numpy from Quickstart ([f9e6555](https://github.com/LEB-EPFL/just-focus/commit/f9e65556c27126156203cb5d1df473ce6a2194fc))

## [1.0.0](https://github.com/LEB-EPFL/just-focus/compare/v0.3.4...v1.0.0) (2026-08-07)


### ⚠ BREAKING CHANGES

* Change waist and beam_center parameter names

### Features

* Add focus_stack script and tests for planes away from z=0 ([#74](https://github.com/LEB-EPFL/just-focus/issues/74)) ([f62996d](https://github.com/LEB-EPFL/just-focus/commit/f62996de3edaa7453978d1923528ca69ef796e23))
* Add mask to the halfmoon pupil calculations ([#72](https://github.com/LEB-EPFL/just-focus/issues/72)) ([33842b3](https://github.com/LEB-EPFL/just-focus/commit/33842b3b7cbf34c5a5a285be23638124a283612f))
* Add pupil phase tilts to model beam steering ([#73](https://github.com/LEB-EPFL/just-focus/issues/73)) ([26b09d4](https://github.com/LEB-EPFL/just-focus/commit/26b09d45cd8f33d413363375c0e77b5395d8f0b3))


### Bug Fixes

* mask_radius_pupil parameter name ([74a8994](https://github.com/LEB-EPFL/just-focus/commit/74a8994219d58fe7290a79da687eb985fcdc63be))


### Documentation

* Add mask_radius_pupil parameter to Halfmoon docs ([b06d2e4](https://github.com/LEB-EPFL/just-focus/commit/b06d2e4a8d6eb32d74e4a265da88fd188d0dc63a))
* Add PyFocus to list of related software ([4e20c4c](https://github.com/LEB-EPFL/just-focus/commit/4e20c4c99a309723d04eaac628d16b0c11188663))
* Fix link to scripts directory in README ([527a204](https://github.com/LEB-EPFL/just-focus/commit/527a204fd461173db5453a1a4ec1ea796b965e59))


### Code Refactoring

* Change waist and beam_center parameter names ([76aee8f](https://github.com/LEB-EPFL/just-focus/commit/76aee8faf330e6d5f2d1e0db4dcd7f54cf9c2177))

## [0.3.4](https://github.com/LEB-EPFL/just-focus/compare/v0.3.3...v0.3.4) (2026-05-26)


### Bug Fixes

* trigger release-please ([49d7764](https://github.com/LEB-EPFL/just-focus/commit/49d7764f0a14a9b49b6a84faa67564c686397ad1))

## [0.3.3](https://github.com/LEB-EPFL/just-focus/compare/v0.3.2...v0.3.3) (2025-11-20)


### Bug Fixes

* Ensure that the origin is sampled in the output coordinate arrays ([#41](https://github.com/LEB-EPFL/just-focus/issues/41)) ([992ee6b](https://github.com/LEB-EPFL/just-focus/commit/992ee6b3246dfdb4115d14f1b1c0216fb1c74900))

## [0.3.2](https://github.com/LEB-EPFL/just-focus/compare/v0.3.1...v0.3.2) (2025-08-07)


### Documentation

* Add PSF-Generator to list of similar packages ([0aa8afd](https://github.com/LEB-EPFL/just-focus/commit/0aa8afd40b760bd9058843cc616cea9b73673c29))
* Improve explanation of the focal in the README ([e553b1e](https://github.com/LEB-EPFL/just-focus/commit/e553b1efcb77e1202484d3d1d1dcd29f85a77736))

## [0.3.1](https://github.com/LEB-EPFL/just-focus/compare/v0.3.0...v0.3.1) (2025-08-06)


### Documentation

* Add description of InputField to README ([29b4dee](https://github.com/LEB-EPFL/just-focus/commit/29b4dee95f2ff36cd7c612917c2c753b59158d94))
* Add propagate and FocalField docs to the README ([04e2118](https://github.com/LEB-EPFL/just-focus/commit/04e211840fda754ff522a78b211043b5b10d9155))
* Add Pupil explanation to the README ([8eb6b61](https://github.com/LEB-EPFL/just-focus/commit/8eb6b61f9ae08b67bd30104df35d31ff1845dc26))
* Add workflow overview to README ([8951c5b](https://github.com/LEB-EPFL/just-focus/commit/8951c5b27dc5b285ebd45e9710d14d0745561092))

## [0.3.0](https://github.com/LEB-EPFL/just-focus/compare/v0.2.2...v0.3.0) (2025-08-05)


### Features

* Add plotting routines for the simulation inputs ([b65cb31](https://github.com/LEB-EPFL/just-focus/commit/b65cb31de953106652f00039ee425d6ec8b54b1a))

## [0.2.2](https://github.com/LEB-EPFL/just-focus/compare/v0.2.1...v0.2.2) (2025-08-05)


### Documentation

* Add installation instructions for plot extra ([#15](https://github.com/LEB-EPFL/just-focus/issues/15)) ([6a2fe14](https://github.com/LEB-EPFL/just-focus/commit/6a2fe1495698fc3a13f31b36f0c1643b532e0bb8))

## [0.2.1](https://github.com/LEB-EPFL/just-focus/compare/v0.2.0...v0.2.1) (2025-08-05)


### Documentation

* Add installation instructions and PyPI badge to README ([#12](https://github.com/LEB-EPFL/just-focus/issues/12)) ([a0633de](https://github.com/LEB-EPFL/just-focus/commit/a0633de399a29052ad177b44cc06d8aa1f272d0c))

## [0.2.0](https://github.com/LEB-EPFL/just-focus/compare/v0.1.0...v0.2.0) (2025-08-05)


### Features

* Add argument to set phase mask center ([#5](https://github.com/LEB-EPFL/just-focus/issues/5)) ([b4ffd5f](https://github.com/LEB-EPFL/just-focus/commit/b4ffd5f5357f20320bce94e58376839a16fcfcc4))

## 0.1.0 (2025-08-04)


### Features

* Add halfmoon focal field script ([7849ae8](https://github.com/LEB-EPFL/just-focus/commit/7849ae8840a4fcce774b789d9030481b2cdae9a9))
* Add linear phase correction and aperture softening ([#2](https://github.com/LEB-EPFL/just-focus/issues/2)) ([4f37e7b](https://github.com/LEB-EPFL/just-focus/commit/4f37e7b3c453b63226be3c32d765ba9e81d46ab6))
* Initial commit ([49390d8](https://github.com/LEB-EPFL/just-focus/commit/49390d810dc5e384826cd62fb4a448ac91c90686))


### Documentation

* Add quickstart to README ([4e624c9](https://github.com/LEB-EPFL/just-focus/commit/4e624c9a7d7afcaea1712b8f209b30d6d367b9c9))
* Add related packages section ([832a94e](https://github.com/LEB-EPFL/just-focus/commit/832a94e1123867406c65a0038dab4ba5fff9b103))
* Add resources to README ([83f2218](https://github.com/LEB-EPFL/just-focus/commit/83f2218b23c19e257134321f8c77f0db688bc05c))
