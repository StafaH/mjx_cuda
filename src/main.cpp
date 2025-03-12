#include <mujoco/mujoco.h>

#include <cstdio>
#include <cstdlib>

#include "device/smooth.h"

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " " \
                << cudaGetErrorString(err) << std::endl;                  \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

int main(int argc, char* argv[]) {
  // print help if arguments are missing
  if (argc < 2 || argc > 6) {
    std::cout << "Usage: " << argv[0] << " <xml_file> <nstep> <batch_size>"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::string xml_file = argv[1];
  int nstep = std::atoi(argv[2]);
  int batch_size = std::atoi(argv[3]);

  std::cout << "Loading MuJoCo model from: " << xml_file << std::endl;
  std::cout << "Using nstep = " << nstep << ", batch_size = " << batch_size
            << std::endl;

  char error[1000];
  mjModel* m = mj_loadXML(xml_file.c_str(), 0, error, 1000);
  if (!m) {
    std::cerr << "Error loading model: " << error << std::endl;
    return EXIT_FAILURE;
  }

  mjData* d = mj_makeData(m);

  // Move model onto GPU.

  /*
  kernel<<<1,1>>>(a);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  */

  // Cleanup
  mj_deleteData(d);
  mj_deleteModel(m);
  
  float elapsed_sec = 1.0f;

  // Calculate performance metrics.
  int total_steps = nstep * batch_size;
  float steps_per_sec = total_steps / elapsed_sec;
  float time_per_step_us = (elapsed_sec / total_steps) * 1e6f;

  // Print benchmark summary.
  std::cout << "Summary for CUDA simulation rollouts (" << total_steps
            << " steps, batch_size=" << batch_size << ")\n\n";
  std::cout << "Total simulation time: " << elapsed_sec << " s\n";
  std::cout << "Total steps per second: " << steps_per_sec << "\n";
  std::cout << "Total time per step: " << time_per_step_us << " µs\n";
  


  // // Allocate device memory and copy the initial state.
  // float* d_data = nullptr;
  // CUDA_CHECK(cudaMalloc(&d_data, total_elements * sizeof(float)));
  // CUDA_CHECK(cudaMemcpy(d_data, h_data, total_elements * sizeof(float),
  //                       cudaMemcpyHostToDevice));

  // // --- Setup CUDA kernel launch parameters ---
  // const int blockSize = 256;
  // const int gridSize = (total_elements + blockSize - 1) / blockSize;
  // const float increment = 0.001f;  // small update per simulation step

  // // Warm up the kernel (to avoid including one-time overhead in timing).
  // simulateKernel<<<gridSize, blockSize>>>(d_data, total_elements, increment);
  // CUDA_CHECK(cudaDeviceSynchronize());

  // // --- Benchmarking loop ---
  // cudaEvent_t start, stop;
  // CUDA_CHECK(cudaEventCreate(&start));
  // CUDA_CHECK(cudaEventCreate(&stop));

  // CUDA_CHECK(cudaEventRecord(start));
  // for (int step = 0; step < nstep; ++step) {
  //   simulateKernel<<<gridSize, blockSize>>>(d_data, total_elements,
  //   increment);
  // }
  // CUDA_CHECK(cudaEventRecord(stop));
  // CUDA_CHECK(cudaEventSynchronize(stop));

  // float elapsed_ms = 0;
  // CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  // float elapsed_sec = elapsed_ms / 1000.0f;

  // // --- Cleanup ---
  // CUDA_CHECK(cudaFree(d_data));
  // delete[] h_data;
  // CUDA_CHECK(cudaEventDestroy(start));
  // CUDA_CHECK(cudaEventDestroy(stop));
  // mj_deleteModel(m);

  return 0;
}

// // Simulation parameters.
// const int nstep = 100;      // Number of simulation steps.
// const int batch_size = 4096; // Number of parallel simulations.
// const float increment = 0.001f;  // Increment value per step.

// // Allocate host memory for initial simulation state.
// float* h_data = new float[batch_size];
// // Initialize the simulation state (for example, all values set to 0.01).
// for (int i = 0; i < batch_size; ++i) {
//     h_data[i] = 0.01f;
// }

// // Allocate device memory.
// float* d_data = nullptr;
// CUDA_CHECK(cudaMalloc(&d_data, batch_size * sizeof(float)));
// CUDA_CHECK(cudaMemcpy(d_data, h_data, batch_size * sizeof(float),
// cudaMemcpyHostToDevice));

// // Set CUDA kernel launch parameters.
// const int blockSize = 256;
// const int gridSize = (batch_size + blockSize - 1) / blockSize;

// // Warm up the kernel to avoid including any first-launch overhead.
// simulateKernel<<<gridSize, blockSize>>>(d_data, batch_size, increment);
// CUDA_CHECK(cudaDeviceSynchronize());

// // Create CUDA events for timing measurement.
// cudaEvent_t start, stop;
// CUDA_CHECK(cudaEventCreate(&start));
// CUDA_CHECK(cudaEventCreate(&stop));

// // Record the start time.
// CUDA_CHECK(cudaEventRecord(start));
// for (int step = 0; step < nstep; ++step) {
//     simulateKernel<<<gridSize, blockSize>>>(d_data, batch_size, increment);
// }
// // Record the stop time.
// CUDA_CHECK(cudaEventRecord(stop));
// CUDA_CHECK(cudaEventSynchronize(stop));

// // Calculate elapsed time in milliseconds.
// float elapsed_ms = 0;
// CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
// float elapsed_sec = elapsed_ms / 1000.0f;

// // Cleanup resources.
// CUDA_CHECK(cudaFree(d_data));
// delete[] h_data;
// CUDA_CHECK(cudaEventDestroy(start));
// CUDA_CHECK(cudaEventDestroy(stop));

// return 0;