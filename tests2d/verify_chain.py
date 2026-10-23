"""Chain family (d = 2, 3). Chain: steady fluence (edge illumination for walls and periodic
against the exact 1D profiles; point-like source against K0 / exp(-mu r)/(4 pi D r)) and
Duhamel constant heating through the coupled integrator. Photothermal: the steady fluence
cases (same function) and uniform constant heating (exact, round-off). Thermoacoustic: the
Duhamel case (same integrator). Smoke, resolution, timing for each class."""
import sys
from _harness import run, main_dims
from pfm2d.solvers import chain as K

if __name__ == "__main__":
    rc = 0
    for d in main_dims(sys.argv):
        va = K.verify_all(d)
        rc |= run(K.Chain(d), verify_result=va)
        pt = {"system": "photothermal", "cases": {
            "light_edge_pressure_release": va["cases"]["light_edge_pressure_release"],
            "light_edge_periodic": va["cases"]["light_edge_periodic"],
            "light_point_source": va["cases"]["light_point_source"],
            "uniform_heating": K.check_uniform_heating(d)},
            "note": "steady fluence cases are the chain's (same steady_fluence function)"}
        pt["passed"] = all(c["passed"] for c in pt["cases"].values())
        rc |= run(K.Photothermal(d), verify_result=pt)
        ta = {"system": "thermoacoustic", "cases": {"constant_heating": va["cases"]["constant_heating"]},
              "note": "the chain's Duhamel case (same run_coupled integrator)"}
        ta["passed"] = va["cases"]["constant_heating"]["passed"]
        rc |= run(K.Thermoacoustic(d), verify_result=ta)
    sys.exit(rc)
