#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <cstring>
#include <cstdlib>
#include <cfloat>

#include "caffe/caffe.hpp"

using namespace caffe;

typedef float Dtype;

void read_file_list(const char* fname, std::vector<std::string>& file_list) {
  std::ifstream infile(fname);
  std::string filename;
  int label;
  while (infile >> filename >> label) {
    std::string basename = filename.substr(0, filename.find_first_of('.'));
    file_list.push_back(basename);
  }
}

void write_predictions_to_file(const char* fname, const std::vector<std::string>& ids, const std::vector<int>& predictions) {
  std::ofstream outfile(fname);
//  CHECK_EQ(ids.size(), predictions.size());

  std::vector<std::string>::const_iterator id_iter = ids.begin();
  std::vector<int>::const_iterator prediction_iter = predictions.begin();

  outfile << "id,label" << std::endl;
  
  for(;prediction_iter != predictions.end(); id_iter++, prediction_iter++) {
      outfile << (*id_iter) << "," << (*prediction_iter) << std::endl;
  }
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 3) {
    LOG(ERROR) << "Usage: predict_net predict_network_proto trained_net predict_filelist";
    return 0;
  }

  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::GPU);

  std::vector<std::string> file_list;
  read_file_list(argv[3], file_list);

  LOG(INFO) << "Predicting " << file_list.size() << " examples";

  std::vector<int> predictions;

  LOG(INFO) << "Loading from " << argv[1];
  
  NetParameter net_param;
  ReadProtoFromTextFile(argv[1], &net_param);

  int batch_size = net_param.layers(0).layer().batchsize();
  int num_examples = file_list.size();
  CHECK_EQ(num_examples % batch_size, 0) << "batch size must divide the number of examples";
  int num_batches = num_examples / batch_size;

  NetParameter trained_net;
  ReadProtoFromBinaryFile(argv[2], &trained_net);

  Net<Dtype> net(net_param);
  net.CopyTrainedLayersFrom(trained_net);

  vector<Blob<Dtype>*> bottom_vec;
  for (int batch=0; batch < num_batches; batch++) {
      LOG(INFO) << "batch " << batch << " of " << num_batches;
      const vector<Blob<Dtype>*>& result =
                  net.Forward(bottom_vec);
      CHECK_EQ(result.size(), 2);
      //result[0] is the fake label from the first layer
      //result[1] are the predicted probabilities

      const Dtype* probabilities = result[1]->cpu_data();

      int dim = result[1]->count() / result[1]->num();
      int num = result[1]->num();

      for (int i = 0; i < num; ++i) {
          Dtype maxval = -FLT_MAX;
          int max_id = 0;
          for (int j = 0; j < dim; ++j) {
            if (probabilities[i * dim + j] > maxval) {
              maxval = probabilities[i * dim + j];
              max_id = j;
            }
          }
          //max_id is the prediction
          predictions.push_back(max_id);
      }
  }

  LOG(INFO) << "Num examples predicted: " << predictions.size();

  write_predictions_to_file("predictions.csv", file_list, predictions);

  return 0;
}
