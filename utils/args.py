"""Command-line options shared by PACT training and evaluation."""

import argparse


def parse(mode):
    parser = argparse.ArgumentParser(
        description=f"PACT {mode}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="dataset/EV-UAV-dataset",
                        help="dataset root containing train, val and test")
    parser.add_argument("--gpu", help="CUDA device index (defaults to the current CUDA visibility)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--input_channel", type=int, default=4)
    parser.add_argument("--whole_t", type=int, default=8000, help="time window in ms")
    parser.add_argument("--res", nargs=2, type=int, default=[352, 288],
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--prediction_thresh", type=float, default=0.9)
    if mode == "train":
        parser.add_argument("--seed", type=int, default=37)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--k", type=int, default=3)
        parser.add_argument("--t", type=int, default=5)
        parser.add_argument("--max_events_num", type=int, default=100000)
        parser.add_argument("--model_save_root", default="checkpoint",
                            help="directory for checkpoints and training metrics")
    else:
        parser.add_argument("--model_path", required=True, help="checkpoint file")
        parser.add_argument("--split", choices=("val", "test"), default="test")
        parser.add_argument("--pd_detT", type=int, default=50, help="Pd/Fa window in ms")
        parser.add_argument("--correct_thresh", type=float, default=0.0001)
        parser.add_argument("--roc", action=argparse.BooleanOptionalAction, default=True,
                            help="compute Pd and Fa")
    cfg = parser.parse_args()
    if not 0 <= cfg.prediction_thresh <= 1:
        parser.error("--prediction_thresh must be between 0 and 1")
    return cfg
