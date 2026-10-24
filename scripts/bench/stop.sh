#!/bin/bash
# stop.sh: stop a running point sequence and any AIPerf process.
pkill -f "[r]un-.*-points.sh"; sleep 1; pkill -INT -f "[a]iperf profile"; sleep 5; pkill -9 -f "[a]iperf"; true
