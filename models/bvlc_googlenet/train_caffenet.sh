#!/usr/bin/env sh

#GLOG_logtostderr=1 ./../../build/tools/caffe train --solver=quick_solver.prototxt 2>&1 | tee googlenet.log
GLOG_logtostderr=1 ./../../build/tools/caffe train --solver=quick_solver.prototxt --snapshot=bvlc_googlenet_quick_iter_600000.solverstate 2>&1 | tee googlenet_part2.log

