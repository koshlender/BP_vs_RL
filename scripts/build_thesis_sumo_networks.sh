#!/usr/bin/env bash
set -euo pipefail
python scripts/generate_thesis_sumo.py
netconvert --node-files sumo/thesis_ch4_ch5/two_intersection/two_intersection.nod.xml \
  --edge-files sumo/thesis_ch4_ch5/two_intersection/two_intersection.edg.xml \
  --tls.guess true \
  --output-file sumo/thesis_ch4_ch5/two_intersection/two_intersection.net.xml
netconvert --node-files sumo/thesis_ch4_ch5/nine_intersection/nine_intersection.nod.xml \
  --edge-files sumo/thesis_ch4_ch5/nine_intersection/nine_intersection.edg.xml \
  --tls.guess true \
  --output-file sumo/thesis_ch4_ch5/nine_intersection/nine_intersection.net.xml
