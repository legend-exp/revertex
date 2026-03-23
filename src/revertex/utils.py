from __future__ import annotations

import logging
import re

import colorlog
import numpy as np
import pyg4ometry.geant4 as pg4
import pygeomhpges
import pygeomtools

from revertex import core

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

    pos = {name: phy_vol_dict[name].position.eval() for name in det_list}

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

    points = core.sample_cylinder(
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
    # logger.addHandler(handler)

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
    """Recursively collect isotopes and their mass fractions for a given material component.

    This function handles components defined as isotopes, elements, or compounds in pyg4ometry.
    For isotopes, it directly adds the ZAID and mass fraction to the `isotopes` dictionary (used as pid in SaG4n).
    For elements, it looks up the natural isotope abundances using the NIST registry.
    For compounds, it recursively processes the sub-components.
    """

    def _to_float(value) -> float:
        if hasattr(value, "eval"):
            return float(value.eval())
        return float(value)

    def _component_mass_weight(
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
            return value
        if kind in {"abundance", "natoms"}:
            return value * _component_reference_mass(
                component,
                nist_registry,
                nist_element_z_to_name,
                pyg4,
            )
        msg = f"Unsupported material component definition kind '{comp_kind}'."
        raise ValueError(msg)

    def _component_reference_mass(
        component,
        nist_registry,
        nist_element_z_to_name: dict[int, str],
        pyg4,
    ) -> float:
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
            nist_element = pyg4.geant4.nist_element_2geant4Element(
                nist_name, nist_registry
            )
            sub_components = nist_element.components

        if not sub_components:
            msg = f"Cannot infer reference mass for component '{getattr(component, 'name', component)}'."
            raise ValueError(msg)

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
        nist_element = pyg4.geant4.nist_element_2geant4Element(nist_name, nist_registry)
        sub_components = nist_element.components

    if not sub_components:
        msg = f"Cannot expand material component '{getattr(component, 'name', component)}' into isotopes."
        raise ValueError(msg)

    comp_values = [
        _component_mass_weight(
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
        total_mass = float(np.sum(list(isotopes.values())))
        if total_mass <= 0:
            msg = "Material component fractions are not valid."
            raise ValueError(msg)
        for zaid in list(isotopes.keys()):
            isotopes[zaid] = isotopes[zaid] / total_mass
