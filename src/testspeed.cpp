#include <mujoco/mujoco.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_profiler_api.h>

#include "device/types.h"
#include "device/io.h"
#include "device/smooth.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
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

  CudaModel* cm = put_model(m);
  CudaData* cd = put_data(m, d, batch_size);
  cudaDeviceSynchronize();

  printf("Data and model transferred to GPU\n");

  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  LaunchKinematicsKernel(stream, batch_size, cm, cd);

  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  printf("Graph instantiated\n");

  cudaGraphLaunch(graphExec, stream);
  cudaStreamSynchronize(stream);
  cudaGraphLaunch(graphExec, stream);
  cudaStreamSynchronize(stream);

  printf("Warmup run completed\n");
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  for (int i = 0; i < nstep; i++) {
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
  }
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double elapsed_sec = milliseconds / 1000.0;
  printf("Kernel execution time: %f ms\n", milliseconds);

  int total_steps = nstep * batch_size;
  int steps_per_sec = total_steps / elapsed_sec;
  double time_per_step_us = (elapsed_sec / total_steps) * 1e6;

  auto fps_string = std::to_string(steps_per_sec);

  int n = fps_string.length() - 3;
  int end = (steps_per_sec >= 0) ? 0 : 1; // Support for negative numbers
  while (n > end) {
    fps_string.insert(n, ",");
    n -= 3;
  }
  printf("Summary for CUDA simulation rollouts (%d steps, batch_size=%d)\n", total_steps, batch_size);
  printf("Total simulation time: %f s\n", elapsed_sec);
  printf("Total steps per second: %s\n", fps_string.c_str());
  printf("Total time per step: %f µs\n", time_per_step_us);

  cudaProfilerStart();
  cudaGraphLaunch(graphExec, stream);
  cudaProfilerStop();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaGraphDestroy(graph);
  cudaGraphExecDestroy(graphExec);
  cudaStreamDestroy(stream);
  
  free_cuda_model(cm);
  free_cuda_data(cd);

  mj_deleteData(d);
  mj_deleteModel(m);

  return EXIT_SUCCESS;
} 