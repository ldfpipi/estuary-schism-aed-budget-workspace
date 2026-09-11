"""
Pioneer River Water LHS-1
Verified SCHISM-native LHS-1 calculation for the Pioneer BudgetE run.

Purpose
-------
Calculate the Main control-volume water-storage change directly from:
    1) hgrid.gr3
    2) Main-CV element list
    3) out2d elevation and dryFlagElement output

The calculation is:

    H_i = d_i + eta_i
    H_e = (H1 + H2 + H3) / 3
    V_e = A_e * H_e            for wet elements
    V_CV = sum(V_e)
    DeltaV = V(T1) - V(T0)

Notes
-----
- hgrid bathymetric depth d is treated as positive-down.
- out2d elevation eta is positive-up.
- zCoordinates is NOT used in this direct calculation; it was used only
  as an independent check of the bathymetry/elevation sign convention.
"""

from pathlib import Path
import math
import re

import numpy as np
from netCDF4 import Dataset


# ======================================================================
# 0. FILES, RUN AND SAVED STATES
# ======================================================================

RUN = Path(
    r"R:\Estuary_SCHISM-Q8213"
    r"\9.0 Developed models"
    r"\Pioneer"
    r"\Pioneer_17_waterlevel_flow_calibration_AprJun2025_BudgetE"
)

HGRID = RUN / "hgrid.gr3"
PARAM = RUN / "param.nml"
CV_FILE = RUN / "cv_pioneer_nested_M4_N1_elements.txt"
OUT = RUN / "outputs"

# Exact saved states used for the 30-day storage calculation.
STATES = {
    "T0": (OUT / "out2d_1.nc", 0, 900.0),
    "T1": (OUT / "out2d_5.nc", 191, 2592000.0),
}

# Horizontal geometry is read from the actual SCHISM runtime settings in
# param.nml. For this Pioneer BudgetE case, ics=2 and the current runtime
# values are expected to be rearth_eq = rearth_pole = 6378206.4 m.


# ======================================================================
# STEP 0A. READ hgrid.gr3
# ======================================================================

def read_grid():
    """
    Read SCHISM horizontal grid.

    Returns
    -------
    title : str
        Grid title.
    lon, lat : ndarray
        1-based node longitude/latitude arrays [degree].
    depth : ndarray
        1-based node bathymetric depth d [m], positive-down.
    elems : list
        1-based element connectivity:
        elems[eid] -> tuple of node IDs.
    """
    with HGRID.open("r", encoding="utf-8", errors="ignore") as f:
        title = f.readline().strip()
        ne, nn = map(int, f.readline().split()[:2])

        # Index 0 is intentionally unused so Python indices equal SCHISM node IDs.
        lon = np.full(nn + 1, np.nan)
        lat = np.full(nn + 1, np.nan)
        depth = np.full(nn + 1, np.nan)

        for _ in range(nn):
            p = f.readline().split()
            nid = int(p[0])
            lon[nid] = float(p[1])
            lat[nid] = float(p[2])
            depth[nid] = float(p[3])

        # Index 0 is intentionally unused so Python indices equal SCHISM element IDs.
        elems = [None] * (ne + 1)

        for _ in range(ne):
            p = f.readline().split()
            eid = int(p[0])
            nv = int(p[1])
            elems[eid] = tuple(map(int, p[2:2 + nv]))

    return title, lon, lat, depth, elems


# ======================================================================
# STEP 0B. READ MAIN-CV ELEMENT IDS
# ======================================================================

def read_main_cv():
    """
    Read the Main control-volume SCHISM element IDs.
    """
    eids = []

    for line in CV_FILE.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        s = line.strip()

        if s and not s.startswith("#"):
            eids.append(int(float(s.split()[0])))

    return np.asarray(sorted(set(eids)), dtype=int)


# ======================================================================
# STEP 3. CALCULATE HORIZONTAL ELEMENT AREA — SCHISM ics=2 NATIVE GEOMETRY
# ======================================================================

def read_param_numeric(text, key):
    """Read one numeric assignment from param.nml."""
    pattern = re.compile(
        rf"(?im)^\s*{re.escape(key)}\s*=\s*"
        rf"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)"
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"Cannot find '{key}' in {PARAM}")
    return float(match.group(1).replace("D", "E").replace("d", "e"))


def read_schism_geometry_settings():
    """
    Read the actual SCHISM horizontal-coordinate settings used by this run.

    This LHS-1 reproduction is specifically for the Pioneer runtime geometry:
        ics = 2  -> longitude/latitude hgrid
        rearth_eq == rearth_pole -> spherical Earth used by SCHISM
    """
    if not PARAM.exists():
        raise RuntimeError(f"Missing param.nml: {PARAM}")

    text = PARAM.read_text(encoding="utf-8", errors="ignore")
    ics = int(round(read_param_numeric(text, "ics")))
    rearth_eq = read_param_numeric(text, "rearth_eq")
    rearth_pole = read_param_numeric(text, "rearth_pole")

    if ics != 2:
        raise RuntimeError(
            f"This script reproduces SCHISM ics=2 lon/lat geometry, but ics={ics}."
        )

    # The actual Pioneer BudgetE run uses equal equatorial/polar radii.
    # Keeping this guard prevents silently claiming an exact reproduction if
    # the runtime geometry is later changed to a true ellipsoid.
    if abs(rearth_eq - rearth_pole) > 1.0e-6:
        raise RuntimeError(
            "This exact Pioneer reproduction currently expects "
            "rearth_eq == rearth_pole.\n"
            f"rearth_eq={rearth_eq}, rearth_pole={rearth_pole}"
        )

    return ics, rearth_eq, rearth_pole


def earth_centred_xyz(lon, lat, rearth_eq, rearth_pole):
    """
    Reproduce the SCHISM ics=2 node mapping from lon/lat [degree] to
    Earth-centred xnd/ynd/znd [m].
    """
    lam = np.deg2rad(lon)
    phi = np.deg2rad(lat)

    xnd = rearth_eq * np.cos(phi) * np.cos(lam)
    ynd = rearth_eq * np.cos(phi) * np.sin(lam)
    znd = rearth_pole * np.sin(phi)

    return xnd, ynd, znd


def schism_native_triangle_area(eid, elems, xnd, ynd, znd, return_details=False):
    """
    Calculate one triangular element area exactly in the SCHISM ics=2 pathway
    used for this spherical Pioneer run:

        node lon/lat
            -> Earth-centred XYZ
            -> element-centre XYZ
            -> element-local east/north frame
            -> xel/yel
            -> SCHISM signa()

    Returns
    -------
    area : float
        Positive horizontal element area [m^2].
    details : dict, optional
        Diagnostic element-centre lon/lat, local xel/yel and signed area.
    """
    n1, n2, n3 = elems[eid]
    nodes = np.asarray([n1, n2, n3], dtype=int)

    # SCHISM element centre in Earth-centred Cartesian coordinates.
    xc = float(np.mean(xnd[nodes]))
    yc = float(np.mean(ynd[nodes]))
    zc = float(np.mean(znd[nodes]))

    # For this actual Pioneer run rearth_eq == rearth_pole, so SCHISM's
    # compute_ll() reduces to the spherical inverse shown below.
    lon_c = math.atan2(yc, xc)
    lat_c = math.atan2(zc, math.hypot(xc, yc))

    # Local zonal (east) axis.
    east = np.asarray([
        -math.sin(lon_c),
        +math.cos(lon_c),
        0.0,
    ])

    # Local meridional (north) axis.
    # SCHISM normalizes this axis explicitly in grid_subs.F90.
    north = np.asarray([
        -math.sin(lat_c) * math.cos(lon_c),
        -math.sin(lat_c) * math.sin(lon_c),
        +math.cos(lat_c),
    ], dtype=float)
    north_norm = float(np.linalg.norm(north))
    if north_norm == 0.0:
        raise RuntimeError(f"Element {eid}: zero-length local north axis.")
    north = north / north_norm

    centre = np.asarray([xc, yc, zc])
    xel = np.zeros(3, dtype=float)
    yel = np.zeros(3, dtype=float)

    for j, nid in enumerate(nodes):
        delta = np.asarray([xnd[nid], ynd[nid], znd[nid]]) - centre
        xel[j] = float(np.dot(delta, east))
        yel[j] = float(np.dot(delta, north))

    # SCHISM signa(): signed planar triangle area in the element-local frame.
    signed_area = 0.5 * (
        (xel[0] - xel[2]) * (yel[1] - yel[2])
        - (xel[1] - xel[2]) * (yel[0] - yel[2])
    )

    # Match SCHISM behaviour: keep the signed area and fail if orientation
    # is non-positive. Do NOT hide an orientation problem with abs().
    if signed_area <= 0.0:
        raise RuntimeError(
            f"Element {eid} has non-positive SCHISM signa() area: "
            f"{signed_area:.16e} m2"
        )

    area = float(signed_area)

    if return_details:
        return area, {
            "nodes": tuple(int(v) for v in nodes),
            "lon_c_deg": math.degrees(lon_c),
            "lat_c_deg": math.degrees(lat_c),
            "xel": xel,
            "yel": yel,
            "signed_area": signed_area,
        }

    return area


# ======================================================================
# READ ONE out2d SAVED STATE
# ======================================================================

def read_state(path, record, expected_time, ne, nn):
    """
    Read time, node elevation and element wet/dry flag for one saved state.

    elevation:
        node-based [m], positive-up

    dryFlagElement:
        element-based
        0 = wet
        1 = dry
    """
    with Dataset(path) as ds:
        time = float(ds.variables["time"][record])

        eta0 = np.ma.filled(
            ds.variables["elevation"][record, :],
            np.nan,
        ).astype(float)

        dry0 = np.ma.filled(
            ds.variables["dryFlagElement"][record, :],
            np.nan,
        ).astype(float)

    if abs(time - expected_time) > 2.0:
        raise RuntimeError(
            f"Unexpected saved time in {path.name}: "
            f"read {time} s, expected {expected_time} s"
        )

    # Convert NetCDF zero-based arrays to 1-based arrays so model IDs can
    # be used directly:
    #     eta[node_id]
    #     dry[element_id]
    eta = np.full(nn + 1, np.nan)
    dry = np.full(ne + 1, np.nan)

    eta[1:] = eta0
    dry[1:] = dry0

    return time, eta, dry


# ======================================================================
# STEPS 1, 2, 4, 5 AND 6
# ELEMENT-BY-ELEMENT MAIN-CV WATER VOLUME
# ======================================================================

def main_cv_volume(cv, elems, depth, area, eta, dry):
    """
    Calculate total Main-CV water volume for one saved state.

    For each wet triangular element:

        STEP 1:
            H_i = d_i + eta_i

        STEP 2:
            H_e = (H1 + H2 + H3) / 3

        STEP 4:
            dryFlagElement determines whether the element contributes.

        STEP 5:
            V_e = A_e * H_e

        STEP 6:
            V_CV = sum(V_e)
    """
    total = 0.0
    wet_count = 0
    dry_count = 0

    for eid in cv:

        # STEP 4: dryFlagElement is element-based.
        # 0 = wet; 1 = dry.
        if dry[eid] not in (0.0, 1.0):
            raise RuntimeError(
                f"Bad dryFlagElement at element {eid}: {dry[eid]}"
            )

        D_e = int(dry[eid])

        if D_e == 1:
            dry_count += 1
            continue

        wet_count += 1

        # Static hgrid connectivity:
        # element ID -> three node IDs belonging to the triangle.
        n1, n2, n3 = elems[eid]

        # STEP 1: nodal water depth H_i = d_i + eta_i [m].
        H1 = depth[n1] + eta[n1]
        H2 = depth[n2] + eta[n2]
        H3 = depth[n3] + eta[n3]

        # STEP 2: triangular element mean water depth H_e [m].
        H_e = (H1 + H2 + H3) / 3.0

        # STEP 5: wet-element water volume V_e [m^3].
        V_e = area[eid] * H_e

        # STEP 6: discrete spatial integration over the Main CV.
        total += V_e

    return total, wet_count, dry_count


# ======================================================================
# REPRESENTATIVE ELEMENT DIAGNOSTIC
# ======================================================================

def print_element_state_diagnostic(eid, elems, depth, area, eta, dry, label):
    """Print the exact element-level quantities used in the LHS-1 inventory."""
    n1, n2, n3 = elems[eid]
    D_e = int(dry[eid])

    H1 = depth[n1] + eta[n1]
    H2 = depth[n2] + eta[n2]
    H3 = depth[n3] + eta[n3]
    H_e = (H1 + H2 + H3) / 3.0
    V_e = 0.0 if D_e == 1 else area[eid] * H_e

    print(f"Element {eid} diagnostic at {label}")
    print(f"  dryFlagElement  = {D_e}")
    print(f"  nodes           = ({n1}, {n2}, {n3})")
    print(
        f"  depth [m]       = {depth[n1]:+.9f}, "
        f"{depth[n2]:+.9f}, {depth[n3]:+.9f}"
    )
    print(
        f"  eta [m]         = {eta[n1]:+.9f}, "
        f"{eta[n2]:+.9f}, {eta[n3]:+.9f}"
    )
    print(
        f"  H1/H2/H3 [m]    = {H1:+.9f}, {H2:+.9f}, {H3:+.9f}"
    )
    print(f"  H_e [m]         = {H_e:+.9f}")
    print(f"  A_e [m2]        = {area[eid]:.12f}")
    print(f"  V_e [m3]        = {V_e:+.9f}")
    print()


# ======================================================================
# STEP 7. CALCULATE T0, T1 AND STORAGE CHANGE
# ======================================================================

def main():
    title, lon, lat, depth, elems = read_grid()
    cv = read_main_cv()

    ne = len(elems) - 1
    nn = len(depth) - 1

    # STEP 3 is time-independent. Read the actual SCHISM runtime geometry
    # and reproduce the ics=2 element-local area calculation.
    ics, rearth_eq, rearth_pole = read_schism_geometry_settings()
    xnd, ynd, znd = earth_centred_xyz(
        lon=lon,
        lat=lat,
        rearth_eq=rearth_eq,
        rearth_pole=rearth_pole,
    )

    area = np.zeros(ne + 1)

    for eid in cv:
        if elems[eid] is None:
            raise RuntimeError(
                f"Element {eid} is missing from hgrid connectivity."
            )

        if len(elems[eid]) != 3:
            raise RuntimeError(
                f"Element {eid} is not a triangle."
            )

        area[eid] = schism_native_triangle_area(
            eid=eid,
            elems=elems,
            xnd=xnd,
            ynd=ynd,
            znd=znd,
        )

    # Representative geometry audit. This does not alter the storage result.
    rep_eid = 149
    if rep_eid in set(cv.tolist()):
        rep_area, rep = schism_native_triangle_area(
            eid=rep_eid,
            elems=elems,
            xnd=xnd,
            ynd=ynd,
            znd=znd,
            return_details=True,
        )
        print("SCHISM horizontal geometry")
        print(f"  ics             = {ics}")
        print(f"  rearth_eq       = {rearth_eq:.6f} m")
        print(f"  rearth_pole     = {rearth_pole:.6f} m")
        print(f"  element 149     = nodes {rep['nodes']}")
        print(f"  centre lon/lat  = {rep['lon_c_deg']:.12f}, {rep['lat_c_deg']:.12f} deg")
        print(f"  signed area     = {rep['signed_area']:.12f} m2")
        for j, nid in enumerate(rep['nodes']):
            print(
                f"  node {nid}: xel={rep['xel'][j]:+.12f} m, "
                f"yel={rep['yel'][j]:+.12f} m"
            )
        print(f"  area(149)       = {rep_area:.12f} m2")
        print()

    results = {}

    for name, (path, record, expected_time) in STATES.items():
        time, eta, dry = read_state(
            path=path,
            record=record,
            expected_time=expected_time,
            ne=ne,
            nn=nn,
        )

        if rep_eid in set(cv.tolist()):
            print_element_state_diagnostic(
                eid=rep_eid,
                elems=elems,
                depth=depth,
                area=area,
                eta=eta,
                dry=dry,
                label=name,
            )

        V, nwet, ndry = main_cv_volume(
            cv=cv,
            elems=elems,
            depth=depth,
            area=area,
            eta=eta,
            dry=dry,
        )

        results[name] = V

        print(f"{name}: time = {time:.0f} s")
        print(f"  wet elements    = {nwet:,}")
        print(f"  dry elements    = {ndry:,}")
        print(f"  Main-CV volume  = {V:,.3f} m3")
        print()

    dV = results["T1"] - results["T0"]

    print("=" * 72)
    print("PIONEER RIVER WATER LHS-1")
    print("=" * 72)
    print(f"hgrid title       = {title}")
    print(f"Main-CV elements  = {len(cv):,}")
    print(f"Main-CV area      = {np.sum(area[cv]):,.6f} m2")
    print(f"V(T0)             = {results['T0']:,.6f} m3")
    print(f"V(T1)             = {results['T1']:,.6f} m3")
    print(f"DeltaV = V1 - V0  = {dV:+,.6f} m3")
    print("=" * 72)


if __name__ == "__main__":
    main()
