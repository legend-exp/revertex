# (alpha,n) Generators

Neutrons from ($\alpha$,n) interactions can produce radioactive isotopes in the
LEGEND detector, such as $^\mathrm{77(m)}$Ge. Compared to the in-situ cosmogenic
production in muon showers, the production via radiogenic neutrons can not
easily be rejected via tagging methods, and thus is important to model
accurately to assess the constraints on the radio-purity of materials close to
the detector.

To model the kinematics of neutrons produced via ($\alpha$,n) interactions from
radiogenic $\alpha$s, vanilla Geant4 is not the most efficient/precise tool.
Thus, dedicated tools have been developed to calculate ($\alpha$,n) yields and
spectra, such as SaG4n \[[1](#ref1)\] and NeuCBOT \[[2](#ref2)\]. This package
offers wrappers for these tools for easy generation of ($\alpha$,n) neutron
kinematics for remage simulations.

> **Warning:** At the moment only a single ($\alpha$,n) generator wrapper is
> available, which is a wrapper around the SaG4n tool. More generators such as
> wrapper around NeuCBOT may be added in the future.

## SaG4n wrapper

SaG4n \[[1](#ref1),[3](#ref3)\] is a tuned Geant4-based tool to calculate
($\alpha$,n) yields and spectra using a combination of evaluated nuclear data
and Talys \[[4](#ref4)\] calculations when the former is unavailable. It
delivers similar results to NeuCBOT \[[2](#ref2),[3](#ref4)\], but since it
properly simulates the reaction, it models all ($\alpha$,Xn) channels with X
being any number of neutrons or gammas from de-excitation of the compound
nucleus.

The revertex ($\alpha$,n) wrapper is an easy way for users to run SaG4n to
generate initial neutron kinematics for ($\alpha$,n) events in the LH5 format
for remage simulations. All generated particles are included in the same event
via `n_part` in the LH5 output.

### 1. Requirements And Setup

#### Runtime Requirements

The wrapper currently expects:

- A container runtime: `docker` or `shifter`.
- Access to a SaG4n container image: `moritzneuberger/sag4n-for-revertex:latest`
  is available on
  [Docker Hub](https://hub.docker.com/r/moritzneuberger/sag4n-for-revertex).

#### Container Setup

If using Docker, ensure the required image is available locally by pulling it
with:

```console
$ docker pull moritzneuberger/sag4n-for-revertex:latest
```

If using Shifter, ensure the image is available in the Shifter cache before
running. You can pull the Docker image into the Shifter repository with:

```console
$ shifterimg pull docker:moritzneuberger/sag4n-for-revertex:latest
```

The wrapper validates this and will provide an error message if the image is
missing. When using an image other than the one above, pass the image name via
`--container-image` in the wrapper input.

### 2. Usage

Users should interact with this wrapper via the CLI subcommand:

```console
$ revertex alpha-n-kin -h
```

The command supports three input pathways.

Common required option:

- `--output-file` for the final LH5 output.

Useful optional options:

- `--n-events` number of simulated primary $\alpha$s (default: `10000000`).
- `--seed` RNG seed (if omitted, a random seed is chosen).
- `--container-runtime` explicit runtime (`docker` or `shifter`).
- `--container-image` SaG4n image (default:
  `moritzneuberger/sag4n-for-revertex:latest`).
- `--output-file-sag4n` path/stem for SaG4n `.out/.root/.log` side outputs.

#### Pathway A: Pre-built SaG4n input file

Provide a complete SaG4n input file with `--input-file-sag4n`.

- Pros: maximum control over SaG4n input.
- Cons: you maintain the full SaG4n card manually.

#### Pathway B: Pre-built material block (sub_material)

Provide `--sub-material` and `--source-chain`, then revertex builds the full
card.

- Pros: avoid writing the full SaG4n template.
- Cons: still manual material definition work.

#### Pathway C: GDML + logical volume part (recommended)

Provide `--gdml-file` and `--part`, plus `--source-chain`, and revertex infers
the isotopic material definition.

- Pros: least manual work and best consistency with detector geometry.
- Cons: requires a suitable GDML geometry with the target logical volume.

This is the recommended pathway for most production cases and new users.

### 3. Examples

#### Example A: Using input_file_sag4n

```console
$ revertex alpha-n-kin \
    --output-file alpha_n_spectrum.lh5 \
    --input-file-sag4n my_sag4n_input.txt \
    --container-runtime docker \
    --container-image moritzneuberger/sag4n-for-revertex:latest
```

#### Example B: Using sub_material + source_chain

```console
$ revertex alpha-n-kin \
    --output-file alpha_n_spectrum.lh5 \
    --sub-material "MATERIAL 1 MyMaterial 1.0 2
32074 0.25
32076 0.75
ENDMATERIAL" \
    --source-chain Th232 \
    --n-events 1000000 \
    --seed 1234567
```

#### Example C: Using GDML + part (recommended)

```console
$ revertex alpha-n-kin \
    --output-file alpha_n_spectrum.lh5 \
    --gdml-file geom.gdml \
    --part V99000A \
    --source-chain U238_lower \
    --container-runtime docker \
    --n-events 2000000
```

## References

(ref1)=

- [1] SaG4n github repository:
  [https://github.com/UIN-CIEMAT/SaG4n](https://github.com/UIN-CIEMAT/SaG4n)

(ref2)=

- [2] NeuCBOT github repository:
  [https://github.com/shawest/neucbot](https://github.com/shawest/neucbot)

(ref3)=

- [3] E. Mendoza et al., Neutron production induced by alpha-decay with Geant4,
  Nucl. Instrum. Methods A 960, 163659 (2020)
  [https://doi.org/10.1016/j.nima.2020.163659](https://doi.org/10.1016/j.nima.2020.163659)

(ref4)=

- [4] Talys project page:
  [https://nds.iaea.org/talys/](https://nds.iaea.org/talys/)
