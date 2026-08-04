"""Apply an overall B-factor to FWT/PHWT map coefficients (Wilson-B matching).

Multiplies the complex map coefficients by the Debye-Waller factor
``exp(-delta_B * d_star_sq / 4)``, where ``delta_B = B_ref - B_dataset``:

  * ``delta_B > 0``  → high-resolution terms damped  → **blurring** (safe).
  * ``delta_B < 0``  → high-resolution terms amplified → **sharpening** (ripple risk).

Run under phenix's python so iotbx/cctbx are importable (do NOT use `uv run`):

    phenix.python scripts/mapvalue/scale_mtz.py IN.mtz OUT.mtz --delta-b <dB> [--labels FWT,PHWT]
    phenix.python scripts/mapvalue/scale_mtz.py IN.mtz OUT.mtz --target-b <B_ref> --source-b <B_i>
    phenix.python scripts/mapvalue/scale_mtz.py IN.mtz OUT.mtz --delta-b 0            # identity check

Optionally also truncate to a common resolution with ``--d-min`` so every map is
built from the same Fourier extent. The output amplitude/phase columns are named
from the first input label (FWT → FWT,PHFWT); pass that pair to
phenix.map_value_at_point afterwards.
"""

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(
        description="Apply an overall delta-B to FWT/PHWT map coefficients."
    )
    p.add_argument("in_mtz", help="Input MTZ (e.g. <pdb>_final.mtz)")
    p.add_argument("out_mtz", help="Output MTZ with scaled coefficients")
    p.add_argument("--labels", default="FWT,PHWT", help="Amplitude,phase labels of the map array")
    p.add_argument(
        "--delta-b",
        type=float,
        default=None,
        help="delta_B = B_ref - B_dataset (Å²); +blur, -sharpen",
    )
    p.add_argument(
        "--target-b",
        type=float,
        default=None,
        help="Reference/target Wilson B (Å²); used with --source-b",
    )
    p.add_argument(
        "--source-b",
        type=float,
        default=None,
        help="This dataset's Wilson B (Å²); used with --target-b",
    )
    p.add_argument(
        "--d-min",
        type=float,
        default=None,
        help="Optional common resolution cutoff (Å) applied after scaling",
    )
    args = p.parse_args()

    if args.delta_b is None:
        if args.target_b is None or args.source_b is None:
            p.error("provide --delta-b, or both --target-b and --source-b")
        delta_b = args.target_b - args.source_b
    else:
        delta_b = args.delta_b

    # Imported here so `--help` works without phenix's python.
    from iotbx import mtz
    from scitbx.array_family import flex

    want = args.labels.split(",")
    arrays = mtz.object(args.in_mtz).as_miller_arrays()
    candidates = [a for a in arrays if want[0] in a.info().labels and a.is_complex_array()]
    if not candidates:
        available = [a.info().labels for a in arrays]
        sys.exit(
            f"No complex array with label {want[0]!r} in {args.in_mtz}. Available: {available}"
        )
    mc = candidates[0]

    # Structure-factor Debye-Waller: F' = F * exp(-delta_B * s²/4), s² = 1/d² = d_star_sq.
    scale = flex.exp(-delta_b * mc.d_star_sq().data() / 4.0)
    scaled = mc.array(data=mc.data() * scale)

    if args.d_min is not None:
        scaled = scaled.resolution_filter(d_min=args.d_min)

    ds = scaled.as_mtz_dataset(column_root_label=want[0])  # writes <want[0]>, PH<want[0]>
    ds.mtz_object().write(args.out_mtz)

    d_lo, d_hi = scaled.d_max_min()
    out_labels = f"{want[0]},PH{want[0]}"
    print(
        f"Wrote {args.out_mtz}  (delta_B={delta_b:+.2f} Å², "
        f"{'blur' if delta_b > 0 else 'sharpen' if delta_b < 0 else 'identity'}, "
        f"d={d_lo:.2f}-{d_hi:.2f} Å)  labels: {out_labels}"
    )


if __name__ == "__main__":
    main()
