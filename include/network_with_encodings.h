
#include "DeviceMem.h"
#include "SwiftNetMLP.h"
#include "encoding.h"
#define WIDTH 64

class NetworkWithEncoding {
 public:
  NetworkWithEncoding(int input_width, int output_width, int n_hidden_layers,
                      Activation activation, Activation output_activation,
                      const int batch_size, int encoding_scale,
                      int encoding_offset);
  ~NetworkWithEncoding() {}

  DeviceMem<float>* forward_pass(GPUMatrix<float>& input,
                                 int use_inference = 0);

  void backward_pass();

  void initialize_params(int use_easy = 0);

  void free_memory();
  sycl::queue get_queue() { return m_q; }

 private:
  Encoding<bf16>* encoding;
  SwiftNetMLP<WIDTH>* network;
  queue m_q;
  GPUMatrix<bf16> encoding_output;

  DeviceMem<bf16> network_input;
  DeviceMem<float> network_output;
};