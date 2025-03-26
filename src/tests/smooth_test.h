#pragma once

#include "test_base.h"
#include "device/smooth.h"
#include "device/io.h"
#include <cmath>

class SmoothTest : public TestBase {
public:
    ~SmoothTest() override {
        if (cuda_model) {
            free_cuda_model(cuda_model);
        }
        if (cuda_data) {
            free_cuda_data(cuda_data);
        }
    }

    void init(mjModel* m, mjData* d, int bs) override {
        model = m;
        data = d;
        batch_size = bs;
    }

    bool run_test() override {
        model->opt.jacobian = mjJAC_SPARSE;
        // give the system a little kick to ensure we have non-identity rotations
        for (int i = 0; i < model->nq; i++) {
            data->qvel[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
        for (int i = 0; i < 3; i++) {
            mj_step(model, data);
        }
        mj_forward(model, data);
        
        cuda_model = put_model(model);
        cudaDeviceSynchronize();
        cuda_data = put_data(model, data, batch_size);
        cudaDeviceSynchronize();

        // Zero out the xanchor xaxis xpos xquat arrays before running the kernel
        cudaMemset(cuda_data->xanchor, 0, model->njnt * sizeof(float) * 3 * batch_size);
        cudaMemset(cuda_data->xaxis, 0, model->njnt * sizeof(float) * 3 * batch_size);
        cudaMemset(cuda_data->xpos, 0, model->nbody * sizeof(float) * 3 * batch_size);
        cudaMemset(cuda_data->xquat, 0, model->nbody * sizeof(float) * 4 * batch_size);

        cudaDeviceSynchronize();

        LaunchKinematicsKernel(cudaStreamDefault, batch_size, cuda_model, cuda_data);

        cudaDeviceSynchronize();

        return compare_results();
    }

    const char* get_name() const override {
        return "kinematics";
    }

private:
    bool compare_results() {
        // Allocate CPU arrays for comparison
        float* cpu_xanchor = new float[model->njnt * 3];
        float* cpu_xaxis = new float[model->njnt * 3];
        float* cpu_xpos = new float[model->nbody * 3];
        float* cpu_xquat = new float[model->nbody * 4];
        float* cpu_xipos = new float[model->nbody * 3];
        float* cpu_xmat = new float[model->nbody * 9];
        float* cpu_ximat = new float[model->nbody * 9];
        float* cpu_geom_xpos = new float[model->ngeom * 3];
        float* cpu_geom_xmat = new float[model->ngeom * 9];
        float* cpu_site_xpos = new float[model->nsite * 3];
        float* cpu_site_xmat = new float[model->nsite * 9];

        // Copy data from device to CPU arrays
        cudaMemcpy(cpu_xanchor, cuda_data->xanchor, model->njnt * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xaxis, cuda_data->xaxis, model->njnt * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xpos, cuda_data->xpos, model->nbody * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xquat, cuda_data->xquat, model->nbody * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xipos, cuda_data->xipos, model->nbody * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xmat, cuda_data->xmat, model->nbody * 9 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_ximat, cuda_data->ximat, model->nbody * 9 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_geom_xpos, cuda_data->geom_xpos, model->ngeom * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_geom_xmat, cuda_data->geom_xmat, model->ngeom * 9 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_site_xpos, cuda_data->site_xpos, model->nsite * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_site_xmat, cuda_data->site_xmat, model->nsite * 9 * sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        const float TOLERANCE = 5e-2f;
        bool passed = true;

        // Compare joint arrays
        for (int i = 0; i < model->njnt && passed; i++) {
            for (int j = 0; j < 3; j++) {
                if (std::abs(cpu_xanchor[i * 3 + j] - data->xanchor[i * 3 + j]) > TOLERANCE ||
                    std::abs(cpu_xaxis[i * 3 + j] - data->xaxis[i * 3 + j]) > TOLERANCE) {
                    std::cerr << "Joint mismatch at " << i << std::endl;
                    passed = false;
                    break;
                }
            }
        }

        // Compare body arrays
        for (int i = 0; i < model->nbody && passed; i++) {
            for (int j = 0; j < 3; j++) {
                if (std::abs(cpu_xpos[i * 3 + j] - data->xpos[i * 3 + j]) > TOLERANCE ||
                    std::abs(cpu_xipos[i * 3 + j] - data->xipos[i * 3 + j]) > TOLERANCE) {
                    std::cerr << "Body position mismatch at " << i << std::endl;
                    passed = false;
                    break;
                }
            }
            for (int j = 0; j < 4 && passed; j++) {
                if (std::abs(cpu_xquat[i * 4 + j] - data->xquat[i * 4 + j]) > TOLERANCE) {
                    std::cerr << "Body quaternion mismatch at " << i << std::endl;
                    passed = false;
                    break;
                }
            }
            for (int j = 0; j < 9 && passed; j++) {
                if (std::abs(cpu_xmat[i * 9 + j] - data->xmat[i * 9 + j]) > TOLERANCE ||
                    std::abs(cpu_ximat[i * 9 + j] - data->ximat[i * 9 + j]) > TOLERANCE) {
                    std::cerr << "Body matrix mismatch at " << i << std::endl;
                    passed = false;
                    break;
                }
            }
        }

        // // Compare geometry arrays
        // for (int i = 0; i < model->ngeom && passed; i++) {
        //     for (int j = 0; j < 3; j++) {
        //         if (std::abs(cpu_geom_xpos[i * 3 + j] - data->geom_xpos[i * 3 + j]) > TOLERANCE) {
        //             std::cerr << "Geom position mismatch at " << i << std::endl;
        //             passed = false;
        //             break;
        //         }
        //     }
        //     for (int j = 0; j < 9 && passed; j++) {
        //         if (std::abs(cpu_geom_xmat[i * 9 + j] - data->geom_xmat[i * 9 + j]) > TOLERANCE) {
        //             std::cerr << "Geom matrix mismatch at " << i << std::endl;
        //             passed = false;
        //             break;
        //         }
        //     }
        // }

        // // Compare site arrays
        // for (int i = 0; i < model->nsite && passed; i++) {
        //     for (int j = 0; j < 3; j++) {
        //         if (std::abs(cpu_site_xpos[i * 3 + j] - data->site_xpos[i * 3 + j]) > TOLERANCE) {
        //             std::cerr << "Site position mismatch at " << i << std::endl;
        //             passed = false;
        //             break;
        //         }
        //     }
        //     for (int j = 0; j < 9 && passed; j++) {
        //         if (std::abs(cpu_site_xmat[i * 9 + j] - data->site_xmat[i * 9 + j]) > TOLERANCE) {
        //             std::cerr << "Site matrix mismatch at " << i << std::endl;
        //             passed = false;
        //             break;
        //         }
        //     }
        // }

        // Free CPU arrays
        delete[] cpu_xanchor;
        delete[] cpu_xaxis;
        delete[] cpu_xpos;
        delete[] cpu_xquat;
        delete[] cpu_xipos;
        delete[] cpu_xmat;
        delete[] cpu_ximat;
        delete[] cpu_geom_xpos;
        delete[] cpu_geom_xmat;
        delete[] cpu_site_xpos;
        delete[] cpu_site_xmat;

        return passed;
    }
}; 