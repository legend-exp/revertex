# Double-beta decay generator

This generator samples the two electrons of double-beta decay, for example the
two-neutrino mode (2νββ) of ⁷⁶Ge. The energies of the two electrons are sampled
from a 2D spectrum given in a file, and their directions from the angular
correlation of the decay. The events are written in the
{ref}`remage kinematics format <remage:manual-input>`, with two electrons per
decay. The position of the decay is not part of the output and is set in
_remage_, for example uniformly in the germanium detectors.

## Input spectrum

The spectrum is a text file with one line per point of a grid of the two
electron kinetic energies $(T_1, T_2)$:

```text
index_1  index_2  T_1  T_2  value
```

This is the format of the `*_2ds.txt` files of Kotila and Iachello
[^kotila2012], which can be downloaded for many isotopes from
[nucleartheory.yale.edu](https://nucleartheory.yale.edu/double-beta-decay-phase-space-factors).
The indices are not used. The energies are in MeV by default (option `--eunit`).
The value is the spectrum $d^2\Gamma/dT_1 dT_2$ at that point, in arbitrary
units; it must not be negative. Points missing from the file, such as those with
$T_1 + T_2$ above the Q-value, are zero. The file can be gzip compressed
(`.txt.gz`).

Each grid point is the centre of a histogram bin. The generator draws a bin with
probability proportional to its value, then $T_1$ and $T_2$ uniformly within the
bin.

## Angular correlation

The direction of the first electron is isotropic. The angle $\theta$ between the
two electrons follows the leading-order approximation for a $0^+ \to 0^+$ decay
[^doi1985]

$$
\frac{dN}{d\cos\theta} \propto 1 - \beta_1 \beta_2 \cos\theta ,
\qquad
\beta_i = \frac{\sqrt{T_i\,(T_i + 2 m_e)}}{T_i + m_e} ,
$$

where $\beta_i$ is the velocity of electron $i$ (in units of $c$) and $m_e$ the
electron mass. The electrons are therefore preferentially emitted back to back.
The factor $\beta_1\beta_2$ uses the same leading-order electron wave functions
in the Coulomb field of the nucleus as the usual Fermi function [^doi1985].
Exact Dirac wave functions with screening make the correlation stronger: for
⁷⁶Ge by about 4 % at electron energies of 500 keV, and by up to 15 % at energies
near 50 keV and 1.5 MeV [^kotila2012].

The generator samples $\cos\theta$ by inverting the cumulative distribution (see
{func}`.double_beta.sample_cos_theta`), and the azimuth of the second electron
around the first uniformly.

## Usage

### Command line

```console
> revertex --seed 42 double-beta-kin \
    --input-file 76Ge_2ds.txt \
    --n-events 1000000 \
    --out-file 76Ge_2vbb.lh5
```

| option               | default | description                                          |
| -------------------- | ------- | ---------------------------------------------------- |
| `--input-file`, `-i` | —       | text file with the 2D spectrum                       |
| `--eunit`, `-e`      | `MeV`   | unit of the energies in the file: `eV`, `keV`, `MeV` |
| `--n-events`, `-n`   | —       | number of decays                                     |
| `--out-file`, `-o`   | —       | output LH5 file                                      |

The seed is the global option `--seed`, given before `double-beta-kin`.

### Python

```python
from revertex import utils
from revertex.generators import double_beta

# write to a file, in chunks
double_beta.save_double_beta_kinematics(
    1_000_000, "76Ge_2ds.txt", "76Ge_2vbb.lh5", seed=42
)

# or get the decays as an awkward array
spectrum = utils.read_input_2d_spectrum("76Ge_2ds.txt")
events = double_beta.generate_double_beta_kinematics(10_000, spectrum=spectrum, seed=1)
```

### remage

The output only contains the kinematics, so the positions are set in the
_remage_ macro, for example:

```text
/RMG/Generator/Select FromFile
/RMG/Generator/FromFile/FileName 76Ge_2vbb.lh5

/RMG/Generator/Confine Volume
/RMG/Generator/Confinement/Physical/AddVolume V.*
```

[^kotila2012]:
    J. Kotila and F. Iachello, Phys. Rev. C 85, 034316 (2012),
    [arXiv:1209.5722](https://arxiv.org/abs/1209.5722).

[^doi1985]:
    M. Doi, T. Kotani and E. Takasugi, Prog. Theor. Phys. Suppl. 83, 1 (1985).
