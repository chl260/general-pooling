#!/usr/bin/env sh

GLOG_logtostderr=1 ./../../build/tools/caffe train --solver=solver.prototxt
