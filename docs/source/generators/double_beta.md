# Double-beta decay generator

This generator samples the two electrons of double-beta decay, for example the
two-neutrino mode (2νββ) of ⁷⁶Ge. The energies of the two electrons are sampled
from a 2D spectrum given in a file, and their directions from an approximation
of the angular correlation of the decay. The events are written in the
{ref}`remage kinematics format <remage:manual-input>`, with two electrons per
decay.

## Format of the input spectrum

The input is the 2D spectrum $d^2\Gamma/dT_1\,dT_2$ of the kinetic energies
$T_1$ and $T_2$ of the two electrons, tabulated on a grid. The format is that of
the `*_2ds.txt` files of Kotila and Iachello [^kotila2012], which can be
downloaded for many isotopes from
[nucleartheory.yale.edu](https://nucleartheory.yale.edu/double-beta-decay-phase-space-factors).
The file must follow these rules; the generator stops with an error otherwise
(see {func}`.utils.validate_2d_spectrum`).

1. It is a text file, optionally gzip compressed (`.txt.gz`). Lines starting
   with `#` are comments.
2. Each line has five columns, separated by spaces:
   `index_1  index_2  T_1  T_2  value`.
3. The indices are positive integers $k_1, k_2$, and the energies are $T_i =
   k_i\,\Delta T$. The step $\Delta T$ is the same for both electrons, so the
   grid is uniform and starts at $\Delta T$.
4. The energies are in MeV, unless another unit is given with `--eunit` (`eV`,
   `keV` or `MeV`).
5. `value` is the spectrum at that grid point, in arbitrary units. It must be a
   finite number and not negative, and at least one value must be positive.
6. Each pair $(k_1, k_2)$ appears at most once. Grid points that are not in the
   file, such as those with $T_1 + T_2$ above the Q-value, are zero.

For example, the first lines of a spectrum with $\Delta T = 1$ keV:

```text
    1     1  0.0010  0.0010   0.0000E+00
    1     2  0.0010  0.0020   0.4534E-19
    1     3  0.0010  0.0030   0.4726E-19
```

Grid point $k$ is the centre of the energy bin $[(k - \tfrac{1}{2})\,\Delta T,\
(k + \tfrac{1}{2})\,\Delta T]$. The generator draws a bin $(k_1, k_2)$ with
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

:::{warning}

This angular correlation is an approximation. The factor $\beta_1\beta_2$ uses
the same leading-order electron wave functions in the Coulomb field of the
nucleus as the usual Fermi function [^doi1985]. Exact Dirac wave functions with
screening make the correlation stronger: for ⁷⁶Ge by about 4 % at electron
energies of 500 keV, and by up to 15 % at energies near 50 keV and 1.5 MeV
[^kotila2012]. For some detectors the correlation may need to be updated to a
more exact one.

:::

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

[^kotila2012]:
    J. Kotila and F. Iachello, Phys. Rev. C 85, 034316 (2012),
    [arXiv:1209.5722](https://arxiv.org/abs/1209.5722).

[^doi1985]:
    M. Doi, T. Kotani and E. Takasugi, Prog. Theor. Phys. Suppl. 83, 1 (1985).
