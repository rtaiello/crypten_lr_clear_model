#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run mpc_cifar example in multiprocess mode:

$ python3 examples/mpc_cifar/launcher.py \
    --evaluate \
    --resume path-to-model/model.pth.tar \
    --batch-size 1 \
    --print-freq 1 \
    --skip-plaintext \
    --multiprocess

To run mpc_cifar example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/mpc_cifar/mpc_cifar.py,\
path-to-model/model.pth.tar\
      examples/mpc_cifar/launcher.py \
      --evaluate \
      --resume model.pth.tar \
      --batch-size 1 \
      --print-freq 1 \
      --skip-plaintext
"""

import argparse
import logging
import os

from src.multiprocess_launcher import MultiProcessLauncher


parser = argparse.ArgumentParser(description="CrypTen Global Model Training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--epochs", default=1, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=100,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=100,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--model-location",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)

parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument("--lr-decay", default=0.1, type=float, help="lr decay factor")
parser.add_argument(
    "--skip-plaintext",
    default=False,
    action="store_true",
    help="Skip validation for plaintext network",
)
parser.add_argument(
    "--multiprocess",
    default=True,
    action="store_true",
    help="Run example in multiprocess mode",
)


def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from src.run import run_crypten_mnist

    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    run_crypten_mnist(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )


def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)