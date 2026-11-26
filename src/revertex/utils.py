from __future__ import annotations

import logging
import re

import colorlog
import hist
import numpy as np
import pyg4ometry.geant4 as pg4
import pygeomhpges
import pygeomtools
from pyg4ometry import geant4

from revertex import sampling

log = logging.getLogger(__name__)


def expand_regex(inputs: list, patterns: list) -> list:
    """Get a list of detectors from regex

    This matches any wildcars with * or ? in the patterns.

    Parameters
    ----------
    inputs
        list of input strings to find matches in.
    patterns
        list of patterns to search for.
    """
    regex_patterns = [
        re.compile(
            "^" + p.replace(".", r"\.").replace("*", ".*").replace("?", ".") + "$"
        )
        for p in patterns
    ]
    return [v for v in inputs if any(r.fullmatch(v) for r in regex_patterns)]


def read_input_beta_csv(path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Reads a CSV file into numpy arrays.

    The file should have the following format:

        energy_1, phase_space_1
        energy_2, phase_space_2
        energy_3, phase_space_3

    Parameters
    ----------
    path
        filepath to the csv file.
    kwargs
        keyword arguments to pass to `np.genfromtxt`
    """
    return np.genfromtxt(path, **kwargs).T[0], np.genfromtxt(path, **kwargs).T[1]


def read_input_2d_spectrum(path: str, *, eunit: str = "MeV") -> hist.Hist:
    r"""Read a 2D spectrum of two particle energies into a histogram.

    The format is that of the double-beta decay spectra of Kotila and Iachello
    (``*_2ds.txt`` at https://nucleartheory.yale.edu). The file must follow
    these rules, otherwise a :class:`ValueError` is raised:

    - Text file, optionally gzip compressed (``.gz``). Lines starting with
      ``#`` are comments.
    - Five whitespace-separated columns per line::

          index_1  index_2  energy_1  energy_2  value

    - The indices are positive integers :math:`k` and the energies are
      :math:`k \, \Delta E`, with the same step :math:`\Delta E > 0` for both
      particles. The grid is therefore uniform and starts at :math:`\Delta E`.
    - Each pair of indices appears at most once.
    - The values are finite and not negative, and at least one is positive.
      They are the spectrum at the grid point, in arbitrary units.

    Grid point :math:`k` becomes the centre of the histogram bin
    :math:`[(k - 1/2)\,\Delta E, (k + 1/2)\,\Delta E]`, for both axes. Grid
    points missing from the file, for example above the endpoint, are zero.

    Parameters
    ----------
    path
        path to the text file.
    eunit
        the unit of the energies in the file: `eV`, `keV` or `MeV`. The
        histogram is in keV.
    """
    factors = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3}
    if eunit not in factors:
        msg = f"Only eunits eV, keV or MeV are supported not {eunit}"
        raise ValueError(msg)

    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 5:
        msg = f"{path} must have 5 columns, not {data.shape[1]}"
        raise ValueError(msg)

    if not np.all(np.isfinite(data)):
        msg = f"{path} contains values that are not finite numbers"
        raise ValueError(msg)

    index = data[:, :2]
    if np.any(index < 1) or np.any(index != np.round(index)):
        msg = f"{path}: the indices must be positive integers"
        raise ValueError(msg)
    index = index.astype(np.int64)

    # the grid step, from the first point, then check all energies against it.
    # The tolerance allows for the rounding of the energies in the file.
    energy = data[:, 2:4] * factors[eunit]
    step = energy[0, 0] / index[0, 0]
    if step <= 0 or not np.allclose(energy, index * step, rtol=0, atol=1e-3 * step):
        msg = (
            f"{path}: the energies must be index * step, on one uniform grid "
            "for both particles"
        )
        raise ValueError(msg)

    value = data[:, 4]
    if np.any(value < 0):
        msg = f"{path} has negative values, which are not a valid spectrum"
        raise ValueError(msg)
    if not np.any(value > 0):
        msg = f"{path} has no positive values, which is not a valid spectrum"
        raise ValueError(msg)

    n_bins = index.max()
    if len(np.unique((index[:, 0] - 1) * n_bins + index[:, 1] - 1)) != len(index):
        msg = f"{path} lists some pairs of indices more than once"
        raise ValueError(msg)

    histo = hist.Hist(
        hist.axis.Regular(n_bins, 0.5 * step, (n_bins + 0.5) * step),
        hist.axis.Regular(n_bins, 0.5 * step, (n_bins + 0.5) * step),
    )
    histo.view()[index[:, 0] - 1, index[:, 1] - 1] = value

    return histo


def find_mother_physical_volumes(
    pv: geant4.PhysicalVolume, registry: geant4.Registry
) -> list[geant4.PhysicalVolume]:
    """Find the mother physical volumes."""
    mothers = []

    for other_pv in registry.physicalVolumeDict.values():
        lv = other_pv.logicalVolume

        if pv in lv.daughterVolumes:
            mothers.append(other_pv)

    return mothers


def _get_position(pv_name: str, reg: geant4.Registry) -> list:
    """Get the global position of a physical volume from the GDML"""

    pv = reg.physicalVolumeDict[pv_name]
    local_pos = pv.position

    n_mothers = 1

    while n_mothers > 0:
        mothers = find_mother_physical_volumes(pv, reg)
        n_mothers = len(mothers)

        if len(mothers) > 1:
            msg = "Only support finding global position if every volume is only placed once."
            raise RuntimeError(msg)
        if len(mothers) == 1:
            if mothers[0].rotation.eval() != [0, 0, 0]:
                msg = "Only support finding global position without rotations"
                raise RuntimeError(msg)

            local_pos += mothers[0].position

            pv = mothers[0]

    return local_pos.eval()


def get_hpges(
    reg: pg4.geant4.registry, detectors: str | list[str]
) -> tuple[dict, dict]:
    """Extract the objects for each HPGe detector in `reg` and in the list of `detectors`"""

    phy_vol_dict = reg.physicalVolumeDict
    det_list = expand_regex(list(phy_vol_dict.keys()), list(detectors))

    hpges = {
        name: pygeomhpges.make_hpge(
            pygeomtools.get_sensvol_metadata(reg, name), registry=None
        )
        for name in det_list
    }

    pos = {name: _get_position(name, reg) for name in det_list}

    return hpges, pos


def get_surface_indices(hpge: pygeomhpges.base.HPGe, surface_type: str | None) -> tuple:
    """Get which surface index corresponds to the desired surface type"""

    surf = np.array(hpge.surfaces)
    return (
        np.where(surf == surface_type)[0]
        if (surface_type is not None)
        else np.arange(len(hpge.surfaces))
    )


def get_surface_weights(hpges: dict, surface_type: str | None) -> list:
    """Get a weighting for each hpge in the `hpges` based on surface area
    for a given `surface_type`
    """

    # index of the surfaces per detector
    surf_ids_tot = [
        np.array(hpge.surfaces) == surface_type
        if surface_type is not None
        else np.arange(len(hpge.surfaces))
        for name, hpge in hpges.items()
    ]

    # total surface area per detector
    surf_tot = [
        np.sum(hpge.surface_area(surf_ids).magnitude)
        for (name, hpge), surf_ids in zip(hpges.items(), surf_ids_tot, strict=True)
    ]

    return surf_tot / np.sum(surf_tot)


def get_borehole_volume(hpge: pygeomhpges.HPGe, size=1000000):
    """Estimate the borehole volume (with MC)"""

    r, z = hpge.get_profile()
    height = max(z)
    radius = max(r)

    points = sampling.sample_cylinder(
        r_range=(0, radius), z_range=(0, height), seed=None, size=size
    )
    vol = np.pi * radius**2 * height

    is_good = len(points[hpge.is_inside_borehole(points)])

    return (is_good / size) * vol


def get_borehole_weights(hpges: dict) -> list:
    """Get a weighting for each hpge in the `hpges` based on borehole volume"""

    vol_tot = [get_borehole_volume(hpge, size=int(1e6)) for _, hpge in hpges.items()]

    return vol_tot / np.sum(vol_tot)


def setup_log(level: int | None = None) -> None:
    """Setup a colored logger for this package.

    Parameters
    ----------
    level
        initial log level, or ``None`` to use the default.
    """
    fmt = "%(log_color)s%(name)s [%(levelname)s]"
    fmt += " %(message)s"

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(fmt))

    logger = logging.getLogger("revertex")
    logger.addHandler(handler)

    if level is not None:
        logger.setLevel(level)


def collect_isotopes(
    component,
    scale: float,
    isotopes: dict[int, float],
    nist_registry,
    nist_element_z_to_name: dict[int, str],
    pyg4,
    *,
    normalize_output: bool = True,
    _is_recursive_call: bool = False,
) -> None:
    """Recursively collect isotopes and their atomic fractions for a given material component.

    This function handles components defined as isotopes, elements, or compounds in pyg4ometry.
    For isotopes, it directly adds the ZAID and atomic fraction to the `isotopes` dictionary (used as pid in SaG4n).
    For elements, it looks up the natural isotope abundances using the NIST registry.
    For compounds, it recursively processes the sub-components.
    """

    def _to_float(value) -> float:
        if hasattr(value, "eval"):
            return float(value.eval())
        return float(value)

    def _resolve_component_reference(component, nist_registry):
        if isinstance(component, str):
            material_dict = getattr(nist_registry, "materialDict", None)
            if isinstance(material_dict, dict) and component in material_dict:
                return material_dict[component]
        return component

    def _component_atomic_weight(
        component,
        comp_value,
        comp_kind: str,
        nist_registry,
        nist_element_z_to_name: dict[int, str],
        pyg4,
    ) -> float:
        value = _to_float(comp_value)
        kind = comp_kind.lower()

        if kind == "massfraction":
            reference_mass = _component_reference_mass(
                component,
                nist_registry,
                nist_element_z_to_name,
                pyg4,
            )
            if reference_mass <= 0:
                msg = "Material component fractions are not valid."
                raise ValueError(msg)
            return value / reference_mass
        if kind in {"abundance", "natoms"}:
            return value
        msg = f"Unsupported material component definition kind '{comp_kind}'."
        raise ValueError(msg)

    def _component_reference_mass(
        component,
        nist_registry,
        nist_element_z_to_name: dict[int, str],
        pyg4,
    ) -> float:
        component = _resolve_component_reference(component, nist_registry)

        if component.__class__.__name__ == "Isotope":
            if hasattr(component, "a"):
                return _to_float(component.a)
            return _to_float(component.N)

        sub_components = getattr(component, "components", None)
        if not sub_components and hasattr(component, "Z"):
            z = round(_to_float(component.Z))
            nist_name = nist_element_z_to_name.get(z)
            if nist_name is None:
                msg = f"Cannot resolve natural isotope abundances for Z={z}."
                raise ValueError(msg)
            # Use a fresh registry to expand NIST element to isotopes.
            # We only need the isotope components, not to store the element.
            temp_registry = pyg4.geant4.Registry()
            nist_element = pyg4.geant4.nist_element_2geant4Element(
                nist_name, temp_registry
            )
            sub_components = nist_element.components

        if not sub_components and hasattr(component, "A"):
            return _to_float(component.A)

        if any(
            str(comp_kind).lower() == "massfraction"
            for _, _, comp_kind in sub_components
        ):
            msg = f"Cannot infer reference mass for component '{getattr(component, 'name', component)}' because it is defined by mass fractions."
            raise ValueError(msg)

        # For abundance/natoms this is the sum over atomic masses weighted by atom counts/fractions.
        return float(
            np.sum(
                [
                    _to_float(comp_value)
                    * _component_reference_mass(
                        sub_component,
                        nist_registry,
                        nist_element_z_to_name,
                        pyg4,
                    )
                    for sub_component, comp_value, _comp_kind in sub_components
                ]
            )
        )

    component = _resolve_component_reference(component, nist_registry)

    if component.__class__.__name__ == "Isotope":
        z = round(_to_float(component.Z))
        a = round(_to_float(component.N))
        zaid = z * 1000 + a
        isotopes[zaid] = isotopes.get(zaid, 0.0) + scale
        return

    sub_components = getattr(component, "components", None)
    if not sub_components and hasattr(component, "Z"):
        z = round(_to_float(component.Z))
        nist_name = nist_element_z_to_name.get(z)
        if nist_name is None:
            msg = f"Cannot resolve natural isotope abundances for Z={z}."
            raise ValueError(msg)
        # Use a fresh registry to expand NIST element to isotopes.
        # We only need the isotope components, not to store the element.
        temp_registry = pyg4.geant4.Registry()
        nist_element = pyg4.geant4.nist_element_2geant4Element(nist_name, temp_registry)
        sub_components = nist_element.components

    if not sub_components and hasattr(component, "Z") and hasattr(component, "A"):
        z = round(_to_float(component.Z))
        a = round(_to_float(component.A))
        zaid = z * 1000 + a
        isotopes[zaid] = isotopes.get(zaid, 0.0) + scale
        return

    if not sub_components:
        msg = f"Cannot expand material component '{getattr(component, 'name', component)}' into isotopes."
        raise ValueError(msg)

    comp_values = [
        _component_atomic_weight(
            sub_component,
            comp_value,
            comp_kind,
            nist_registry,
            nist_element_z_to_name,
            pyg4,
        )
        for sub_component, comp_value, comp_kind in sub_components
    ]
    total = float(np.sum(comp_values))
    if total <= 0:
        msg = "Material component fractions are not valid."
        raise ValueError(msg)

    for (sub_component, _comp_value, _comp_kind), weighted_value in zip(
        sub_components, comp_values, strict=True
    ):
        collect_isotopes(
            sub_component,
            scale * weighted_value / total,
            isotopes,
            nist_registry,
            nist_element_z_to_name,
            pyg4,
            normalize_output=normalize_output,
            _is_recursive_call=True,
        )

    if normalize_output and not _is_recursive_call:
        total_atoms = float(np.sum(list(isotopes.values())))
        if total_atoms <= 0:
            msg = "Material component fractions are not valid."
            raise ValueError(msg)
        for zaid in list(isotopes.keys()):
            isotopes[zaid] = isotopes[zaid] / total_atoms
