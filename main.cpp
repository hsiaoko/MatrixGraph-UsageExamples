#include <cuda_runtime.h>
#include <torch/script.h>
#include <execution>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <utility>
#include <memory>

#include "MatrixGraph/core/matrixgraph.cuh"
#include "MatrixGraph/core/task/gpu_task/matrix_ops.cuh"
#include "MatrixGraph/core/task/gpu_task/task_base.cuh"
#include "MatrixGraph/core/common/types.h"
#include "MatrixGraph/core/common/yaml_config.h"
#include "MatrixGraph/core/components/scheduler/scheduler.h"
#include "core/task/gpu_task/matrix_ops.cuh"

// Type aliases for MatrixGraph components
using sics::matrixgraph::core::components::scheduler::SchedulerType;
using sics::matrixgraph::core::task::MatrixOps;
using UnifiedOwnedBufferFloat =
    sics::matrixgraph::core::data_structures::UnifiedOwnedBuffer<float>;
using BufferFloat = sics::matrixgraph::core::data_structures::Buffer<float>;

/**
 * @brief Converts a PyTorch tensor to a CPU float buffer
 * @param tensor Input PyTorch tensor (must be float type)
 * @return Pointer to newly allocated buffer containing tensor data
 * @note The caller is responsible for freeing the returned buffer
 */
static float* ConvertTensor2Buffer(const torch::Tensor& tensor) {
  float* tensor_buf =
      new float[tensor.numel()]();  // Allocate zero-initialized buffer
  std::memcpy(tensor_buf, tensor.data_ptr<float>(),  // Copy tensor data
              sizeof(float) * tensor.numel());
  return tensor_buf;
}

/**
 * @brief Extracts parameters from a TorchScript module into CPU buffers
 * @param module Loaded TorchScript module
 * @return Array of pointers to parameter buffers
 * @note Prints parameter values for debugging
 * @warning Caller must free both the array and each parameter buffer
 */
float** ConvertModule2Buffers(const torch::jit::Module& module) {
  float** output_buffers_ptr = new float*[module.parameters().size()]();

  int i = 0;
  for (const auto& param : module.parameters()) {
    auto contig_param = param.contiguous();  // Ensure memory is contiguous
    output_buffers_ptr[i] = new float[contig_param.numel()]();
    memcpy(output_buffers_ptr[i], contig_param.data_ptr<float>(),
           sizeof(float) * contig_param.numel());

    // Debug print parameter values
    std::cout << "Layer: ";
    for (int i = 0; i < contig_param.numel(); i++) {
      std::cout << contig_param.data_ptr<float>()[i] << " ";
    }
    std::cout << std::endl;
    i++;
  }
  return output_buffers_ptr;
}

/**
 * @brief Performs model inference using MatrixGraph operations
 * @param input Input data buffer (size must match model expectations)
 * @param buffers_ptr Array of model parameter buffers
 * @param task MatrixOps instance for GPU operations
 * @return Unified buffer containing output (managed memory)
 */
UnifiedOwnedBufferFloat Model(float* input,
                              float** buffers_ptr,
                              MatrixOps& task) {
  // Initialize dummy input (all ones)
  for (int i = 0; i < 10; i++) {
    input[i] = 1.0f;
  }

  // Matrix dimensions (hardcoded for example)
  int m = 1;   // Rows in A
  int k = 10;  // Columns in A / Rows in B
  int n = 2;   // Columns in B

  // CPU-side buffers
  BufferFloat buf_A, buf_B, buf_C, buf_D;

  // Configure input buffer
  buf_A.data = input;
  buf_A.size = sizeof(float) * m * k;

  // Configure weight buffer (first parameter)
  buf_B.data = buffers_ptr[0];
  buf_B.size = sizeof(float) * k * n;

  // Allocate output buffer
  buf_C.data = new float[m * n]();
  buf_C.size = sizeof(float) * m * n;

  // Configure bias buffer (second parameter)
  buf_D.data = buffers_ptr[1];
  buf_D.size = sizeof(float) * m * n;

  // Debug print input matrices
  std::cout << "A" << std::endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      std::cout << buf_A.data[i * k + j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "B" << std::endl;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < k; i++) {
      std::cout << buf_B.data[i + j * k] << " ";
    }
    std::cout << std::endl;
  }

  // Unified memory buffers for GPU operations
  UnifiedOwnedBufferFloat unified_buf_A, unified_buf_B, unified_buf_C,
      unified_buf_D;

  // Initialize unified buffers
  unified_buf_A.Init(buf_A);
  unified_buf_B.Init(buf_B);
  unified_buf_D.Init(buf_D);
  unified_buf_C.Init(sizeof(float) * m * n);  // Output buffer

  // Perform operations on GPU
  task.MatMult(unified_buf_A.GetPtr(), unified_buf_B.GetPtr(),
               unified_buf_C.GetPtr(), m, k, n);  // Matrix multiplication
  task.MatAdd(unified_buf_D.GetPtr(), unified_buf_C.GetPtr(), m,
              n);                               // Add bias
  task.Activate(unified_buf_C.GetPtr(), m, n);  // Activation function

  // Print final output
  std::cout << "OUTPUT:" << std::endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << unified_buf_C.GetPtr()[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

/**
 * @brief Main function demonstrating model inference pipeline
 * @note Workflow:
 * 1. Load traced PyTorch model
 * 2. Convert parameters to buffers
 * 3. Perform inference using MatrixGraph operations
 */
int main(int argc, char* argv[]) {
  // Create random tensor and convert to buffer
  torch::Tensor tensor = torch::rand({2, 10});
  float* data_buf = ConvertTensor2Buffer(tensor);

  // Model parameter buffers
  float** buffers;
  std::cout << "LoadModule" << std::endl;

  try {
    // Load traced TorchScript model
    auto module = torch::jit::load(
        "/home/zhuxk/projects/use_matrixgraph/models/full_model_traced.pt");

    // Extract model parameters
    buffers = ConvertModule2Buffers(module);

  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model: " << e.what() << "\n";
    return -1;
  }

  try {
    auto* task = new MatrixOps();  // Create GPU task handler

    // Run inference
    Model(data_buf, buffers, *task);

    delete task;  // Clean up

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
