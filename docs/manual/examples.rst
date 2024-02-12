.. Copyright (C) 2024 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause


==========
Benchmarks
==========

We include several benchmarks in the reposity for the Inference and the Training in the 
benchmark directory. Note that the benchmarks only include minimal tests to ensure correctness of the results.
Correctness tests are part of the unit tests, which can be found in the test directory. 

Running the tests requires either a PVC GPU in the computer system (if built with -DTARGET_DEVICE="PVC") 
or an ARC GPU (-DTARGET_DEVICE="ARC").

========
Examples
========

