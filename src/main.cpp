#include <mujoco/mujoco.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_profiler_api.h>

#include "device/types.h"
#include "device/io.h"
#include "device/smooth.h"


int main(int argc, char* argv[]) {
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

  // Transfer model and data to GPU
  CudaModel* cm = put_model(m);
  CudaData* cd = put_data(m, d, batch_size);

  // Launch the kernel
  dim3 blockSize(1024);
  dim3 gridSize((batch_size * m->nq + blockSize.x - 1) / blockSize.x);

  // Create CUDA graph objects
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  // Capture the kernel launch into a CUDA graph
  cudaStreamBeginCapture(cudaStreamDefault, cudaStreamCaptureModeGlobal);

  LaunchKinematicsKernel(batch_size, cm, cd);

  cudaStreamEndCapture(cudaStreamDefault, &graph);
  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  // Warmup run
  cudaGraphLaunch(graphExec, cudaStreamDefault);

  // Test against CPU Mujoco
  m->opt.jacobian = mjJAC_SPARSE;

  mj_step(m, d);
  mj_forward(m, d);
  
  // Copy CUDA data back to CPU for comparison
  CudaData* cpu_cd = new CudaData();
  
  // Allocate and copy arrays for body data
  cpu_cd->xanchor = new float[m->nbody];
  cpu_cd->xaxis = new float[m->nbody];
  cpu_cd->xpos = new float[m->nbody];
  cpu_cd->xquat = new float[m->nbody];
  cpu_cd->xipos = new float[m->nbody];
  cpu_cd->xmat = new float[m->nbody * 9];
  cpu_cd->ximat = new float[m->nbody * 9];
  
  cudaMemcpy(cpu_cd->xanchor, cd->xanchor, m->nbody * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_cd->xaxis, cd->xaxis, m->nbody * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_cd->xpos, cd->xpos, m->nbody * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_cd->xquat, cd->xquat, m->nbody * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_cd->xipos, cd->xipos, m->nbody * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_cd->xmat, cd->xmat, m->nbody * 9 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_cd->ximat, cd->ximat, m->nbody * 9 * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Allocate and copy arrays for geometry data
  cpu_cd->geom_xpos = new float[m->ngeom];
  cpu_cd->geom_xmat = new float[m->ngeom * 9];
  
  cudaMemcpy(cpu_cd->geom_xpos, cd->geom_xpos, m->ngeom * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_cd->geom_xmat, cd->geom_xmat, m->ngeom * 9 * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Allocate and copy arrays for site data
  cpu_cd->site_xpos = new float[m->nsite];
  cpu_cd->site_xmat = new float[m->nsite * 9];
  
  cudaMemcpy(cpu_cd->site_xpos, cd->site_xpos, m->nsite * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_cd->site_xmat, cd->site_xmat, m->nsite * 9 * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Define tolerance for floating point comparisons
  const float TOLERANCE = 5e-5f;
  
  // Macro for comparing values with tolerance
  #define ASSERT_CLOSE(cuda_val, cpu_val, name) \
    if (std::abs(cuda_val - cpu_val) > TOLERANCE) { \
      std::cerr << "Error: " << name << " mismatch. CUDA: " << cuda_val \
                << ", CPU: " << cpu_val << std::endl; \
      return EXIT_FAILURE; \
    }
  
  std::cout << "Comparing arrays" << std::endl;
  for (int i = 0; i < m->nbody; i++) {
    ASSERT_CLOSE(cpu_cd->xanchor[i], d->xanchor[i], "xanchor");
    ASSERT_CLOSE(cpu_cd->xaxis[i], d->xaxis[i], "xaxis");
    ASSERT_CLOSE(cpu_cd->xpos[i], d->xpos[i], "xpos");
    ASSERT_CLOSE(cpu_cd->xquat[i], d->xquat[i], "xquat");
    ASSERT_CLOSE(cpu_cd->xipos[i], d->xipos[i], "xipos");
  }
  
  std::cout << "Comparing matrices" << std::endl;
  for (int i = 0; i < m->nbody; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        ASSERT_CLOSE(cpu_cd->xmat[i * 9 + j * 3 + k], d->xmat[i * 9 + j * 3 + k], "xmat");
        ASSERT_CLOSE(cpu_cd->ximat[i * 9 + j * 3 + k], d->ximat[i * 9 + j * 3 + k], "ximat");
      }
    }
  }
  
  std::cout << "Comparing geometry arrays" << std::endl;
  for (int i = 0; i < m->ngeom; i++) {
    ASSERT_CLOSE(cpu_cd->geom_xpos[i], d->geom_xpos[i], "geom_xpos");
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        ASSERT_CLOSE(cpu_cd->geom_xmat[i * 9 + j * 3 + k], d->geom_xmat[i * 9 + j * 3 + k], "geom_xmat");
      }
    }
  }
  
  std::cout << "Comparing site arrays" << std::endl;
  for (int i = 0; i < m->nsite; i++) {
    ASSERT_CLOSE(cpu_cd->site_xpos[i], d->site_xpos[i], "site_xpos");
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        ASSERT_CLOSE(cpu_cd->site_xmat[i * 9 + j * 3 + k], d->site_xmat[i * 9 + j * 3 + k], "site_xmat");
      }
    }
  }
  
  std::cout << "All CUDA outputs match CPU outputs within tolerance of " << TOLERANCE << std::endl;
  
  // Clean up the CPU copy
  delete[] cpu_cd->xanchor;
  delete[] cpu_cd->xaxis;
  delete[] cpu_cd->xpos;
  delete[] cpu_cd->xquat;
  delete[] cpu_cd->xipos;
  delete[] cpu_cd->xmat;
  delete[] cpu_cd->ximat;
  delete[] cpu_cd->geom_xpos;
  delete[] cpu_cd->geom_xmat;
  delete[] cpu_cd->site_xpos;
  delete[] cpu_cd->site_xmat;
  delete cpu_cd;
  
  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  // Execute the graph nstep times
  for (int i = 0; i < nstep; i++) {
    cudaGraphLaunch(graphExec, cudaStreamDefault);
  }
  
  // Record stop event and synchronize
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float elapsed_sec = milliseconds / 1000.0f;
  printf("Kernel execution time: %f ms\n", milliseconds);

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

  // Use CUDA profiler markers for additional profiling
  cudaProfilerStart();
  cudaGraphLaunch(graphExec, cudaStreamDefault);
  cudaProfilerStop();

  // Cleanup timing events and graph resources
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaGraphDestroy(graph);
  cudaGraphExecDestroy(graphExec);

  // Cleanup CUDA resources
  free_cuda_model(cm);
  free_cuda_data(cd);

  // Cleanup MuJoCo resources
  mj_deleteData(d);
  mj_deleteModel(m);

  return EXIT_SUCCESS;
}