# Third-Party Software

MPQ downloads the following projects during installation. They are not
relicensed by MPQ and remain subject to their upstream licenses.

| Project | Pinned revision | License |
| --- | --- | --- |
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | `ed96f2e04ec868b04f969733a242ce3b5bb7cb1e` | Apache-2.0 |
| [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) | `e7706faf8d1c3b9f241e36860640ad1dac644ede` | BSD-3-Clause |
| [QUIP#](https://github.com/Cornell-RelaxML/quip-sharp) | `1d8f873e9a2a8b86b12bb1064c312c5689b77d98` | GPL-3.0 |

The corresponding upstream license files are present in each checkout after
running `install.sh`. Compatibility patches under `patches/` are intended for
use with the named upstream project. In particular, use and redistribution of
the QUIP# backend and its patch remain subject to GPL-3.0.

Model weights and datasets are not distributed with MPQ. Their respective
licenses and terms apply separately.
