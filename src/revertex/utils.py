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
    """Read a 2D spectrum of two particle energies into a histogram.

    The file is a whitespace-separated text file, optionally gzip compressed
    (``.gz``), in the format of the double-beta decay spectra of Kotila and
    Iachello (``*_2ds.txt`` at https://nucleartheory.yale.edu)::

        index_1  index_2  energy_1  energy_2  value

    The two indices are not used. The energies of each axis are the values that
    appear in the file. They form a grid, and each grid point becomes the centre
    of a histogram bin. The value is the spectrum at that point, in arbitrary
    units. Grid points missing from the file, for example above the endpoint,
    are set to zero.

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

    e1 = data[:, 2] * factors[eunit]
    e2 = data[:, 3] * factors[eunit]
    value = data[:, 4]

    if np.any(value < 0):
        msg = f"{path} has negative values, which are not a valid spectrum"
        raise ValueError(msg)

    points_1 = np.unique(e1)
    points_2 = np.unique(e2)

    histo = hist.Hist(
        hist.axis.Variable(_bin_edges(points_1)),
        hist.axis.Variable(_bin_edges(points_2)),
    )
    histo.view()[np.searchsorted(points_1, e1), np.searchsorted(points_2, e2)] = value

    return histo


def _bin_edges(points: np.ndarray) -> np.ndarray:
    """Bin edges with the grid `points` at the bin centres, not below zero."""
    if len(points) < 2:
        msg = "the spectrum needs at least two energies per axis"
        raise ValueError(msg)

    mid = (points[1:] + points[:-1]) / 2
    first = max(points[0] - (mid[0] - points[0]), 0)
    last = points[-1] + (points[-1] - mid[-1])

    return np.concatenate(([first], mid, [last]))


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
