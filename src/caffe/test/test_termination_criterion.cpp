// Copyright 2014 Tobias Domhan

#include <stdlib.h>

#include <ctime>

#include <cstring>
#include <algorithm>

#include "gtest/gtest.h"
#include "caffe/common.hpp"

#include "caffe/proto/caffe.pb.h"

#include "caffe/test/test_caffe_main.hpp"

#include "caffe/net.hpp"
#include "caffe/solver.hpp"


namespace caffe {

  typedef double Dtype;

  TEST(TestTerminationCriterion, MaxIter) {
    MaxIterTerminationCriterion<Dtype> criterion(3);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyIteration(1);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyIteration(2);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyIteration(3);
    EXPECT_TRUE(criterion.IsCriterionMet());
  }
  
  TEST(TestTerminationCriterion, TestAccuracy) {
    TestAccuracyTerminationCriterion<Dtype> criterion(3);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());

    //first countdown
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //second countdown
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //reset
    criterion.NotifyTestAccuracy(0.6);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //first countdown
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //second countdown
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //third countdown
    criterion.NotifyTestAccuracy(0.5);
    
    EXPECT_TRUE(criterion.IsCriterionMet());
  }

  TEST(TestTerminationCriterion, ExternalRunInBackgroundTerminationCriterion) {
    int run_every = 10;
    int ret;
    ExternalRunInBackgroundTerminationCriterion<Dtype> criterion("touch test", run_every);

    EXPECT_FALSE(criterion.IsCriterionMet());

    criterion.NotifyTestAccuracy(0.5);
    EXPECT_TRUE(std::ifstream("learning_curve.txt"));

    criterion.NotifyIteration(run_every+1);
    EXPECT_TRUE(std::ifstream("termination_criterion_running"));
    criterion.NotifyIteration(run_every+2);
    EXPECT_FALSE(criterion.IsCriterionMet());

    ret = system("rm termination_criterion_running");

    criterion.NotifyIteration(run_every+3);
    EXPECT_FALSE(criterion.IsCriterionMet());

    criterion.NotifyIteration(2*run_every+1);

    ret = system("rm termination_criterion_running");
    ret = system("touch y_predict.txt");

    criterion.NotifyIteration(2*run_every+2);

    EXPECT_TRUE(criterion.IsCriterionMet());

    //make sure the touch was run:
    EXPECT_TRUE(std::ifstream("test"));
    ret = system("rm test");
  }


  TEST(TestTerminationCriterion, ExternalRunInBackgroundTerminationCriterionIsRunInBackground) {
    int run_every = 10;
    double epsilon_time = 1.;
    int ret;
    ExternalRunInBackgroundTerminationCriterion<Dtype> criterion("sleep 5", run_every);

    //check that the command is actually run in the background.

    clock_t begin = clock();
    criterion.NotifyIteration(run_every+1);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    EXPECT_TRUE(elapsed_secs < epsilon_time);
    LOG(INFO) << elapsed_secs;
  }

  TEST(TestTerminationCriterion, ExternalRunInBackgroundTerminationCriterionIsRun) {
    int run_every = 10;
    int ret;
    ExternalRunInBackgroundTerminationCriterion<Dtype> criterion("touch test", run_every);

    //check that the command is actually run in the background.

    criterion.NotifyIteration(run_every+1);
    sleep(1);
    EXPECT_TRUE(std::ifstream("test"));
    ret = system("rm test");
  }

}  // namespace caffe
