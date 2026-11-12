#!/bin/bash
# MoS2 on HfO2, 2808 atoms (device_MoS2HfO2_2808a_29088o): the release workflow with its defaults (steps A-E + plots,
# slab cut 1e-8, tile cut 1e-6, DE 0.05,0.15,0.2). Kept for 0.2.x users; see examples/workflow.sh for the options.
#   examples/mos2_hfo2_workflow.sh DEVICE_FOLDER [OUT]          (GPU=1 for --gpu)
exec "$(dirname "$0")/workflow.sh" "${1:?usage: $0 DEVICE_FOLDER [OUT]}" "${2:-mos2_hfo2_workflow}"
