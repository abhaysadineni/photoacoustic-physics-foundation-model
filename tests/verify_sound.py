"""Sound (uniform, layered, lossy): exact checks, smoke tests and timing for all three
classes. Writes results/verification/sound.json; exits non-zero if any class fails."""
import sys

from _tissue_harness import environment, print_rep, report_and_exit, run

from pfm.solvers import radial, sound

BCS = ("periodic", "pressure_release", "rigid")

if __name__ == "__main__":
    out, ok_all = {"system": "sound"}, True
    for cls in (sound.SoundUniform, sound.SoundLayered, sound.SoundLossy):
        rep, ok = run(cls(), BCS)
        print_rep(rep)
        for r in rep["smoke_test"]["rows"]:
            if r["failures"]:
                print(f"  {rep['system']} smoke {r['bc']}: {r['failures']}")
        for r in rep["timing_64x64"]:
            print(f"  {rep['system']} timing {r['bc']}: {r['seconds']:.2f} s for 64 x 64")
        out[rep["system"]] = rep
        ok_all &= ok
    out["dt_frame_s"] = sound.DT_FRAME
    out["length_m"] = sound.LENGTH
    out["f_ref_Hz"] = sound.F_REF
    out["param_ranges"] = sound.PARAM_RANGES
    out["param_sources"] = sound.PARAM_SOURCES
    out["wave_spectral_radius_dr_over_c_n512"] = {
        f"d{d}_{bc}": radial.wave_spectral_radius(512, d, bc)
        for d in (1, 2, 3) for bc in (BCS if d == 1 else BCS[1:])}
    out["passed"] = bool(ok_all)
    out["environment"] = environment()
    from pfm import common
    path = common.RESULTS / "verification" / "sound.json"
    common.save_json(out, path)
    print(f"  sound: all classes passed={ok_all}; wrote {path}")
    sys.exit(0 if ok_all else 1)
