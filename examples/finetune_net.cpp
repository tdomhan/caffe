// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly finetune a network.
// Usage:
//    finetune_net solver_proto_file pretrained_net

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>

#include "caffe/caffe.hpp"

using namespace caffe;

typedef float Dtype;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(ERROR) << "Usage: finetune_net solver_proto_file pretrained_net";
    return 0;
  }

  if(getenv("CAFFE_DEVICE_ID")) {
    Caffe::SetDevice(atoi(getenv("CAFFE_DEVICE_ID")));
  }

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[1], &solver_param);

  LOG(INFO) << "Starting Optimization";
  SGDSolver<Dtype> solver(solver_param);
  LOG(INFO) << "Loading from " << argv[2];
  solver.net()->CopyTrainedLayersFrom(string(argv[2]));
  solver.Solve();
  LOG(INFO) << "Optimization Done.";

  LOG(INFO) << "Accuracy: " << solver.GetBestTestPerformance();

  return 0;
}
