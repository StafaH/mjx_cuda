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
        if (cuda_model == nullptr) {
            printf("Failed to allocate cuda_model\n");
            return false;
        }
        cudaDeviceSynchronize();

        cuda_data = put_data(model, data, batch_size);
        if (cuda_data == nullptr) {
            printf("Failed to allocate cuda_data\n");
            return false;
        }

        cudaDeviceSynchronize();

        // Zero out the xanchor xaxis xpos xquat arrays before running the kernel
        cudaMemset(cuda_data->xanchor, 0, model->njnt * sizeof(vec3p) * batch_size);
        cudaMemset(cuda_data->xaxis, 0, model->njnt * sizeof(vec3p) * batch_size);
        cudaMemset(cuda_data->xpos, 0, model->nbody * sizeof(vec3p) * batch_size);
        cudaMemset(cuda_data->xquat, 0, model->nbody * sizeof(quat) * batch_size);

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
        vec3p* cpu_xanchor = new vec3p[model->njnt];
        vec3p* cpu_xaxis = new vec3p[model->njnt];
        vec3p* cpu_xpos = new vec3p[model->nbody];
        quat* cpu_xquat = new quat[model->nbody];
        vec3p* cpu_xipos = new vec3p[model->nbody];
        mat3p* cpu_xmat = new mat3p[model->nbody];
        mat3p* cpu_ximat = new mat3p[model->nbody];
        vec3p* cpu_geom_xpos = new vec3p[model->ngeom];
        mat3p* cpu_geom_xmat = new mat3p[model->ngeom];
        vec3p* cpu_site_xpos = new vec3p[model->nsite];
        mat3p* cpu_site_xmat = new mat3p[model->nsite];

        // Copy data from device to CPU arrays
        cudaMemcpy(cpu_xanchor, cuda_data->xanchor, model->njnt * sizeof(vec3p), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xaxis, cuda_data->xaxis, model->njnt * sizeof(vec3p), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xpos, cuda_data->xpos, model->nbody * sizeof(vec3p), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xquat, cuda_data->xquat, model->nbody * sizeof(quat), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xipos, cuda_data->xipos, model->nbody * sizeof(vec3p), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xmat, cuda_data->xmat, model->nbody * sizeof(mat3p), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_ximat, cuda_data->ximat, model->nbody * sizeof(mat3p), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_geom_xpos, cuda_data->geom_xpos, model->ngeom * sizeof(vec3p), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_geom_xmat, cuda_data->geom_xmat, model->ngeom * sizeof(mat3p), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_site_xpos, cuda_data->site_xpos, model->nsite * sizeof(vec3p), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_site_xmat, cuda_data->site_xmat, model->nsite * sizeof(mat3p), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        const float TOLERANCE = 5e-2f;
        bool passed = true;

        // Compare joint arrays
        for (int i = 0; i < model->njnt && passed; i++) {
            if (std::abs(cpu_xanchor[i].x - data->xanchor[i * 3]) > TOLERANCE ||
                std::abs(cpu_xanchor[i].y - data->xanchor[i * 3 + 1]) > TOLERANCE ||
                std::abs(cpu_xanchor[i].z - data->xanchor[i * 3 + 2]) > TOLERANCE) {
                std::cerr << "xanchor mismatch at joint " << i << std::endl;
                passed = false;
                break;
            }
            if (std::abs(cpu_xaxis[i].x - data->xaxis[i * 3]) > TOLERANCE ||
                std::abs(cpu_xaxis[i].y - data->xaxis[i * 3 + 1]) > TOLERANCE ||
                std::abs(cpu_xaxis[i].z - data->xaxis[i * 3 + 2]) > TOLERANCE) {
                std::cerr << "xaxis mismatch at joint " << i << std::endl;
                passed = false;
                break;
            }
        }

        // Compare body arrays
        for (int i = 0; i < model->nbody && passed; i++) {
            if (std::abs(cpu_xpos[i].x - data->xpos[i * 3]) > TOLERANCE ||
                std::abs(cpu_xpos[i].y - data->xpos[i * 3 + 1]) > TOLERANCE ||
                std::abs(cpu_xpos[i].z - data->xpos[i * 3 + 2]) > TOLERANCE) {
                std::cerr << "xpos mismatch at body " << i << std::endl;
                passed = false;
                break;
            }
            if (std::abs(cpu_xquat[i].w - data->xquat[i * 4]) > TOLERANCE ||
                std::abs(cpu_xquat[i].x - data->xquat[i * 4 + 1]) > TOLERANCE ||
                std::abs(cpu_xquat[i].y - data->xquat[i * 4 + 2]) > TOLERANCE ||
                std::abs(cpu_xquat[i].z - data->xquat[i * 4 + 3]) > TOLERANCE) {
                std::cerr << "xquat mismatch at body " << i << std::endl;
                passed = false;
                break;
            }
            if (std::abs(cpu_xipos[i].x - data->xipos[i * 3]) > TOLERANCE ||
                std::abs(cpu_xipos[i].y - data->xipos[i * 3 + 1]) > TOLERANCE ||
                std::abs(cpu_xipos[i].z - data->xipos[i * 3 + 2]) > TOLERANCE) {
                std::cerr << "xipos mismatch at body " << i << std::endl;
                passed = false;
                break;
            }

            for (int j = 0; j < 9 && passed; j++) {
                if (std::abs(cpu_xmat[i].m[j] - data->xmat[i * 9 + j]) > TOLERANCE ||
                    std::abs(cpu_ximat[i].m[j] - data->ximat[i * 9 + j]) > TOLERANCE) {
                    std::cerr << "xmat mismatch at body " << i << std::endl;
                    passed = false;
                    break;
                }
            }
        }

        // Compare geometry arrays
        for (int i = 0; i < model->ngeom && passed; i++) {
            if (std::abs(cpu_geom_xpos[i].x - data->geom_xpos[i * 3]) > TOLERANCE ||
                std::abs(cpu_geom_xpos[i].y - data->geom_xpos[i * 3 + 1]) > TOLERANCE ||
                std::abs(cpu_geom_xpos[i].z - data->geom_xpos[i * 3 + 2]) > TOLERANCE) {
                std::cerr << "geom_xpos mismatch at geom " << i << std::endl;
                passed = false;
                break;
            }
            for (int j = 0; j < 9 && passed; j++) {
                if (std::abs(cpu_geom_xmat[i].m[j] - data->geom_xmat[i * 9 + j]) > TOLERANCE) {
                    std::cerr << "geom_xmat mismatch at geom " << i << std::endl;
                    passed = false;
                    break;
                }
            }
        }

        // Compare site arrays
        for (int i = 0; i < model->nsite && passed; i++) {
            if (std::abs(cpu_site_xpos[i].x - data->site_xpos[i * 3]) > TOLERANCE ||
                std::abs(cpu_site_xpos[i].y - data->site_xpos[i * 3 + 1]) > TOLERANCE ||
                std::abs(cpu_site_xpos[i].z - data->site_xpos[i * 3 + 2]) > TOLERANCE) {
                std::cerr << "site_xpos mismatch at site " << i << std::endl;
                passed = false;
                break;
            }
            for (int j = 0; j < 9 && passed; j++) {
                if (std::abs(cpu_site_xmat[i].m[j] - data->site_xmat[i * 9 + j]) > TOLERANCE) {
                    std::cerr << "site_xmat mismatch at site " << i << std::endl;
                    passed = false;
                    break;
                }
            }
        }

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