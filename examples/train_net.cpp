// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>

#include <cstring>

#include <iostream>

#include "caffe/caffe.hpp"

using namespace caffe;

typedef float Dtype;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(ERROR) << "Usage: train_net solver_proto_file [resume_point_file]";
    return 0;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[1], &solver_param);

  LOG(INFO) << "Starting Optimization";
  SGDSolver<Dtype> solver(solver_param);
  if (argc == 3) {
    LOG(INFO) << "Resuming from " << argv[2];
    solver.Solve(argv[2]);
  } else {
    solver.Solve();
  }
  LOG(INFO) << "Optimization Done.";

  std::cout << "Accuracy: " << solver.GetBestTestPerformance();

  return 0;
}
