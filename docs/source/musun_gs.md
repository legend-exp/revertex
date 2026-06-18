# Atmospheric muon generator (musun-gs)

[MUSUN](https://linkinghub.elsevier.com/retrieve/pii/S0010465508003640) is a
Fortran code that samples single atmospheric muons underground based on muon
propagation code. musun-gs is a specific version generated for the **G**ran
**S**asso underground laboratory (LNGS) fit to LVD data. The source code can be
obtained by contacting the author of the above paper.

The _revertex_ interface runs musun-gs inside a container (Docker or Shifter),
parses the ASCII output, and saves the result to an LH5 file following the
{ref}`remage vtx specification <remage:manual-input>`.

## Prerequisites

A pre-built container image of musun-gs must be available. Users can install it
either by pulling the repository and building the container locally:

```console
$ docker pull ghcr.io/legend-exp/musun-gs:latest
```

or (**recommended**) by pulling the image from the GitHub Container Registry
(GHCR):

```console
$ docker login ghcr.io (provide username and personal access token with read:packages scope)
$ docker pull ghcr.io/legend-exp/musun-gs:latest
```

When using Shifter, the container can be pulled and run with the following
commands:

```console
$ shifterimg login ghcr.io (provide username and personal access token with read:packages scope)
$ shifterimg pull ghcr.io/legend-exp/musun-gs:latest
```

## Usage

### Command line

```console
$ revertex musun-gs \
    --out-file muons.lh5 \
    --n-events 1000000
```

All geometry parameters are optional and default to the LNGS Hall A volume:

```console
$ revertex musun-gs \
    --out-file muons.lh5 \
    --n-events 5000000 \
    --seed 42 \
    --dx-cm 4000 \
    --dy-cm 2000 \
    --dz-cm 3500 \
    --container-image ghcr.io/legend-exp/musun-gs:latest
```

Full option reference:

```console
$ revertex musun-gs --help
```

### Python API

```python
from revertex.generators.musun_gs import generate_musun_primaries

generate_musun_primaries(
    n_muons=1_000_000,
    out_file="muons.lh5",
    seed=42,
    dx_cm=4000.0,  # 40 m along x
    dy_cm=2000.0,  # 20 m along y
    dz_cm=3500.0,  # 35 m along z
)
```

## Geometry and coordinate system

Muons are sampled on the surface of a rectangular parallelepiped (cuboid)
**centred at the origin**. The three parameters `dx_cm`, `dy_cm`, `dz_cm` are
the **full widths** of the cuboid in each direction (not half-widths).

The axes follow the LVD coordinate system used internally by musun-gs:

- **x** — short side of Hall A (south-west direction)
- **y** — long side of Hall A (south-east direction)
- **z** — vertical (upward)

Default values of musun-gs reproduce the LNGS Hall A geometry (40 × 20 × 35 m³).

## Output format

The generator writes a single unified table to the LH5 file:

`vtx/kin` : Kinematic and position properties of each muon.

| field    | unit  | description                                   |
| -------- | ----- | --------------------------------------------- |
| `ekin`   | keV   | kinetic energy                                |
| `px`     | keV/c | x-component of momentum                       |
| `py`     | keV/c | y-component of momentum                       |
| `pz`     | keV/c | z-component of momentum                       |
| `time`   | ns    | always 0                                      |
| `g4_pid` | —     | PDG code: −13 (μ⁺) or 13 (μ⁻)                 |
| `n_part` | —     | always 1 (one muon per primary)               |
| `xloc`   | mm    | x-coordinate of entry point on cuboid surface |
| `yloc`   | mm    | y-coordinate of entry point on cuboid surface |
| `zloc`   | mm    | z-coordinate of entry point on cuboid surface |

`vtx/rate` : Scalar. Global muon intensity for the configured geometry as
reported by musun-gs at startup, in units of `(s)^-1`. Useful for normalising
simulation results.

## Chunking and reproducibility

If more than 1 × 10⁶ muons are requested, the generator runs the container
multiple times (chunks of 1 × 10⁶). Each chunk receives a different seed
(`seed × 7ⁱ` for chunk _i_), following the same convention as all other
_revertex_ generators.

## API reference

```{eval-rst}
.. autofunction:: revertex.generators.musun_gs.generate_musun_primaries
   :no-index:
```
