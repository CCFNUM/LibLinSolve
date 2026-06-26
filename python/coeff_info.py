#!/usr/bin/env python3
# File       : coeff_info.py
# Created    : Fri Jun 26 2026 08:55:30 (+0200)
# Author     : Fabian Wermelinger
# Description: Parse the metadata of a serialized coefficients file
# Copyright 2026 CCFNUM HSLU T&A. All Rights Reserved.

import argparse
import numpy as np


def parse_args(*, partial=False):
    parser = argparse.ArgumentParser(
        description="Parse metadata info in serialized coefficients file")
    # yapf: disable
    parser.add_argument("filename", type=str, help="Filename of serialized coefficients")
    # yapf: enable
    if partial:
        return parser.parse_known_args()
    else:
        return parser.parse_args()


def main(args):
    with open(args.filename, 'rb') as blob:
        index_bytes, n_owned_nodes, row_ptr_size, indices_size = np.fromfile(
            blob, dtype=np.uint64, count=4)
        # in bytes (2 = primary + secondary)
        offset = (row_ptr_size + 2 * indices_size) * 8
        blocksize, lhs_size, rhs_size = np.fromfile(blob,
                                                    dtype=np.uint64,
                                                    offset=offset,
                                                    count=3)
        lhs = np.fromfile(blob, dtype=np.double, count=lhs_size)
        rhs = np.fromfile(blob, dtype=np.double, count=rhs_size)

        print(lhs[:10])
        print(rhs[:10])

        assert rhs_size == n_owned_nodes
        assert lhs_size == indices_size
        assert row_ptr_size == n_owned_nodes + 1

        print(f"""Filename:      {args.filename}
Index size:    {index_bytes * 8} bit
Blocksize:     {blocksize}
N owned nodes: {n_owned_nodes}
nnz:           {indices_size}

stats:
rhs: mean: {rhs.mean():e}
      min: {rhs.min():e}
      std: {rhs.std():e}
      max: {rhs.max():e}
       L2: {np.linalg.norm(rhs):e}

lhs: mean: {lhs.mean():e}
      min: {lhs.min():e}
      std: {lhs.std():e}
      max: {lhs.max():e}
       L2: {np.linalg.norm(lhs):e}""")


if __name__ == "__main__":
    args = parse_args()
    main(args)
