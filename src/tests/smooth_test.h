#pragma once

#include "test_base.h"
#include "device/smooth.h"
#include "device/io.h"
#include <cmath>

class SmoothTest : public TestBase {
public:
    ~SmoothTest() override {
        if (cpu_data) {
            cleanup_cpu_data();
            delete cpu_data;
        }
        
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

        cpu_data = new CudaData();
        allocate_cpu_arrays();
    }

    bool run_test() override {
        model->opt.jacobian = mjJAC_SPARSE;
        // give the system a little kick to ensure we have non-identity rotations
        for (int i = 0; i < model->nq; i++) {
            data->qvel[i] = (float)rand() / RAND_MAX * 0.02f - 0.01f;
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

        LaunchKinematicsKernel(batch_size, cuda_model, cuda_data);

        cudaDeviceSynchronize();

        copy_data_to_cpu();

        cudaDeviceSynchronize();

        return compare_results();
    }

    const char* get_name() const override {
        return "kinematics";
    }

private:
    void allocate_cpu_arrays() {
        cpu_data->xanchor = new float[model->njnt * 3];
        cpu_data->xaxis = new float[model->njnt * 3];
        cpu_data->xpos = new float[model->nbody * 3];
        cpu_data->xquat = new float[model->nbody * 4];
        cpu_data->xipos = new float[model->nbody * 3];
        cpu_data->xmat = new float[model->nbody * 9];
        cpu_data->ximat = new float[model->nbody * 9];
        cpu_data->geom_xpos = new float[model->ngeom * 3];
        cpu_data->geom_xmat = new float[model->ngeom * 9];
        cpu_data->site_xpos = new float[model->nsite * 3];
        cpu_data->site_xmat = new float[model->nsite * 9];
    }

    void cleanup_cpu_data() {
        delete[] cpu_data->xanchor;
        delete[] cpu_data->xaxis;
        delete[] cpu_data->xpos;
        delete[] cpu_data->xquat;
        delete[] cpu_data->xipos;
        delete[] cpu_data->xmat;
        delete[] cpu_data->ximat;
        delete[] cpu_data->geom_xpos;
        delete[] cpu_data->geom_xmat;
        delete[] cpu_data->site_xpos;
        delete[] cpu_data->site_xmat;
    }

    void copy_data_to_cpu() {
        cudaMemcpy(cpu_data->xanchor, cuda_data->xanchor, model->njnt * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->xaxis, cuda_data->xaxis, model->njnt * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->xpos, cuda_data->xpos, model->nbody * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->xquat, cuda_data->xquat, model->nbody * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->xipos, cuda_data->xipos, model->nbody * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->xmat, cuda_data->xmat, model->nbody * 9 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->ximat, cuda_data->ximat, model->nbody * 9 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->geom_xpos, cuda_data->geom_xpos, model->ngeom * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->geom_xmat, cuda_data->geom_xmat, model->ngeom * 9 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->site_xpos, cuda_data->site_xpos, model->nsite * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_data->site_xmat, cuda_data->site_xmat, model->nsite * 9 * sizeof(float), cudaMemcpyDeviceToHost);
    }

    bool compare_results() {
        const float TOLERANCE = 5e-2f;
        // print xanchor
        for (int i = 0; i < model->njnt; i++) {
            std::cout << "xanchor[" << i << "]: " 
                      << cpu_data->xanchor[i * 3] << ", "
                      << cpu_data->xanchor[i * 3 + 1] << ", "
                      << cpu_data->xanchor[i * 3 + 2] << std::endl;
        }

        // Compare joint arrays
        for (int i = 0; i < model->njnt; i++) {
            for (int j = 0; j < 3; j++) {
                if (!compare_value(cpu_data->xanchor[i * 3 + j], data->xanchor[i * 3 + j], "xanchor") ||
                    !compare_value(cpu_data->xaxis[i * 3 + j], data->xaxis[i * 3 + j], "xaxis")) {
                    std::cout << "Failed at xanchor[" << i << "][" << j << "]" << std::endl;
                    return false;
                }
            }
        }

        // Compare body arrays
        for (int i = 0; i < model->nbody; i++) {
            for (int j = 0; j < 3; j++) {
                if (!compare_value(cpu_data->xpos[i * 3 + j], data->xpos[i * 3 + j], "xpos") ||
                    !compare_value(cpu_data->xipos[i * 3 + j], data->xipos[i * 3 + j], "xipos")) {
                    return false;
                }
            }
            for (int j = 0; j < 4; j++) {
                if (!compare_value(cpu_data->xquat[i * 4 + j], data->xquat[i * 4 + j], "xquat")) {
                    return false;
                }
            }
        }
        
        // Compare matrices
        for (int i = 0; i < model->nbody; i++) {
            for (int j = 0; j < 9; j++) {
                if (!compare_value(cpu_data->xmat[i * 9 + j], data->xmat[i * 9 + j], "xmat") ||
                    !compare_value(cpu_data->ximat[i * 9 + j], data->ximat[i * 9 + j], "ximat")) {
                    return false;
                }
            }
        }
        
        // Compare geometry arrays
        for (int i = 0; i < model->ngeom; i++) {
            for (int j = 0; j < 3; j++) {
                if (!compare_value(cpu_data->geom_xpos[i * 3 + j], data->geom_xpos[i * 3 + j], "geom_xpos")) {
                    return false;
                }
            }
            for (int j = 0; j < 9; j++) {
                if (!compare_value(cpu_data->geom_xmat[i * 9 + j], data->geom_xmat[i * 9 + j], "geom_xmat")) {
                    return false;
                }
            }
        }
        
        // Compare site arrays
        for (int i = 0; i < model->nsite; i++) {
            for (int j = 0; j < 3; j++) {
                if (!compare_value(cpu_data->site_xpos[i * 3 + j], data->site_xpos[i * 3 + j], "site_xpos")) {
                    return false;
                }
            }
            for (int j = 0; j < 9; j++) {
                if (!compare_value(cpu_data->site_xmat[i * 9 + j], data->site_xmat[i * 9 + j], "site_xmat")) {
                    return false;
                }
            }
        }
        
        return true;
    }

    bool compare_value(float cuda_val, float cpu_val, const char* name) {
        const float TOLERANCE = 5e-2f;
        if (std::abs(cuda_val - cpu_val) > TOLERANCE) {
            std::cerr << "Error: " << name << " mismatch. CUDA: " << cuda_val
                      << ", CPU: " << cpu_val << std::endl;
            return false;
        }
        return true;
    }

    CudaData* cpu_data = nullptr;
}; 