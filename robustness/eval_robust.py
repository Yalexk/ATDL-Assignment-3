"""Evaluate one checkpoint under explicit degradation parameters.

    python -m robustness.eval_robust --checkpoint CKPT --sam_ckpt SAM --data_path ISIC \
        --condition noise:0.05 --condition spots:random:12 --out results/adhoc.csv

The clean condition is always included. For the calibrated study use robustness.run_sweep.
"""
import argparse

from .cli import (add_data_args, add_model_args, add_spot_args, load_model, make_dataset, make_loader,
                  parse_condition, spot_config, write_csv)
from .evaluate import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    add_data_args(parser)
    add_spot_args(parser)
    parser.add_argument('--condition', action='append', default=[], type=parse_condition,
                        help='clean | noise:SIGMA | spots:PLACEMENT:RADIUS (repeatable)')
    parser.add_argument('--out', required=True, help='output CSV, one row per image and condition')
    a = parser.parse_args(argv)

    net = load_model(a)
    loader = make_loader(make_dataset(a), a)
    rows = evaluate(net, loader, a.condition, a.device, a.out_size, a.multimask_output,
                    degrade_seed=a.degrade_seed, spot_cfg=spot_config(a))
    write_csv(rows, a.out)


if __name__ == '__main__':
    main()
