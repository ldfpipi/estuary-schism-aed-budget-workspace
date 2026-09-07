"""
Pioneer River Water LHS-1
Direct Step 0-7 calculation used in the accompanying walkthrough report.

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

import numpy as np
from netCDF4 import Dataset


# ======================================================================
# 0. FILES, RUN AND SAVED STATES
# ======================================================================

RUN = Path(
    r"R:\Estuary_SCHISM-Q8213\9.0 Developed models\Pioneer"
    r"\Pioneer_17_waterlevel_flow_calibration_AprJun2025_BudgetE"
)

HGRID = RUN / "hgrid.gr3"
CV_FILE = RUN / "cv_pioneer_nested_M4_N1_elements.txt"
OUT = RUN / "outputs"

# Exact saved states used for the 30-day storage calculation.
STATES = {
    "T0": (OUT / "out2d_1.nc", 0, 900.0),
    "T1": (OUT / "out2d_5.nc", 191, 2592000.0),
}

# Mean Earth radius [m].
# This matches the current production water-budget geometry.
R_EARTH = 6371008.8


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
# STEP 3. CALCULATE HORIZONTAL ELEMENT AREA
# ======================================================================

def metric_xy(lon, lat):
    """
    Convert longitude/latitude to local Cartesian x/y coordinates [m]
    using a local equirectangular approximation.
    """
    lon0 = np.nanmean(lon[1:])
    lat0 = np.nanmean(lat[1:])

    x = (
        R_EARTH
        * math.cos(math.radians(lat0))
        * np.deg2rad(lon - lon0)
    )
    y = R_EARTH * np.deg2rad(lat - lat0)

    return x, y


def triangle_area(eid, elems, x, y):
    """
    Calculate horizontal area A_e [m^2] for one triangular element.
    """
    n1, n2, n3 = elems[eid]

    return 0.5 * abs(
        (x[n2] - x[n1]) * (y[n3] - y[n1])
        - (x[n3] - x[n1]) * (y[n2] - y[n1])
    )


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
# STEP 7. CALCULATE T0, T1 AND STORAGE CHANGE
# ======================================================================

def main():
    title, lon, lat, depth, elems = read_grid()
    cv = read_main_cv()

    ne = len(elems) - 1
    nn = len(depth) - 1

    # STEP 3 is time-independent:
    # calculate Main-CV element areas once.
    x, y = metric_xy(lon, lat)

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

        area[eid] = triangle_area(eid, elems, x, y)

    results = {}

    for name, (path, record, expected_time) in STATES.items():
        time, eta, dry = read_state(
            path=path,
            record=record,
            expected_time=expected_time,
            ne=ne,
            nn=nn,
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
    print(f"Main-CV area      = {np.sum(area[cv]):,.3f} m2")
    print(f"V(T0)             = {results['T0']:,.3f} m3")
    print(f"V(T1)             = {results['T1']:,.3f} m3")
    print(f"DeltaV = V1 - V0  = {dV:+,.3f} m3")
    print("=" * 72)


if __name__ == "__main__":
    main()
