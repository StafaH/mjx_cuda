#include "io.h"
#include <vector>
#include <algorithm>

CudaModel* put_model(const mjModel* m) {
    CudaModel* cm = new CudaModel();

    cm->nq = m->nq;
    cm->njnt = m->njnt;
    cm->nbody = m->nbody;
    cm->ngeom = m->ngeom;   
    cm->nsite = m->nsite;
    cm->nmocap = m->nmocap;

    std::vector<int> body_depth(m->nbody, -1);
    std::vector<std::vector<int>> bodies;
    body_depth[0] = 0;

    for (int i = 1; i < m->nbody; i++) {
        body_depth[i] = body_depth[m->body_parentid[i]] + 1;
    }
    
    int max_depth = *std::max_element(body_depth.begin(), body_depth.end());

    bodies.resize(max_depth + 1);
    for (int i = 0; i < m->nbody; i++) {
        bodies[body_depth[i]].push_back(i);
    }
    
    std::vector<int> body_tree;
    for (const auto& level_bodies : bodies) {
        body_tree.insert(body_tree.end(), level_bodies.begin(), level_bodies.end());
    }
    
    std::vector<int> body_treeadr;
    int offset = 0;
    for (const auto& level_bodies : bodies) {
        body_treeadr.push_back(offset);
        offset += level_bodies.size();
    }

    cm->nlevel = max_depth + 1;
    cm->nbody_treeadr = body_treeadr.size();
    cudaMalloc(&cm->body_tree, body_tree.size() * sizeof(int));
    cudaMallocHost(&cm->body_treeadr, body_treeadr.size() * sizeof(int));

    cudaMemcpy(cm->body_tree, body_tree.data(), body_tree.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_treeadr, body_treeadr.data(), body_treeadr.size() * sizeof(int), cudaMemcpyHostToHost);

    // Allocate temporary float arrays for conversion
    std::vector<float> qpos0_float(m->nq);
    std::vector<float> body_pos_float(m->nbody * 3);
    std::vector<float> body_quat_float(m->nbody * 4);
    std::vector<float> body_ipos_float(m->nbody * 3);
    std::vector<float> body_iquat_float(m->nbody * 4);
    std::vector<float> jnt_axis_float(m->njnt * 3);
    std::vector<float> jnt_pos_float(m->njnt * 3);
    std::vector<float> geom_pos_float(m->ngeom * 3);
    std::vector<float> geom_quat_float(m->ngeom * 4);
    std::vector<float> site_pos_float(m->nsite * 3);
    std::vector<float> site_quat_float(m->nsite * 4);

    // Convert double arrays to float
    for (int i = 0; i < m->nq; i++) {
        qpos0_float[i] = static_cast<float>(m->qpos0[i]);
    }
    for (int i = 0; i < m->nbody * 3; i++) {
        body_pos_float[i] = static_cast<float>(m->body_pos[i]);
        body_ipos_float[i] = static_cast<float>(m->body_ipos[i]);
    }
    for (int i = 0; i < m->nbody * 4; i++) {
        body_quat_float[i] = static_cast<float>(m->body_quat[i]);
        body_iquat_float[i] = static_cast<float>(m->body_iquat[i]);
    }
    for (int i = 0; i < m->njnt * 3; i++) {
        jnt_axis_float[i] = static_cast<float>(m->jnt_axis[i]);
        jnt_pos_float[i] = static_cast<float>(m->jnt_pos[i]);
    }
    for (int i = 0; i < m->ngeom * 3; i++) {
        geom_pos_float[i] = static_cast<float>(m->geom_pos[i]);
    }
    for (int i = 0; i < m->ngeom * 4; i++) {
        geom_quat_float[i] = static_cast<float>(m->geom_quat[i]);
    }
    for (int i = 0; i < m->nsite * 3; i++) {
        site_pos_float[i] = static_cast<float>(m->site_pos[i]);
    }
    for (int i = 0; i < m->nsite * 4; i++) {
        site_quat_float[i] = static_cast<float>(m->site_quat[i]);
    }

    cudaMalloc(&cm->qpos0, m->nq * sizeof(float));
    cudaMalloc(&cm->body_jntadr, m->njnt * sizeof(int));
    cudaMalloc(&cm->body_jntnum, m->njnt * sizeof(int));
    cudaMalloc(&cm->body_parentid, m->nbody * sizeof(int));
    cudaMalloc(&cm->body_mocapid, m->nbody * sizeof(int));
    cudaMalloc(&cm->body_pos, m->nbody * 3 * sizeof(float));
    cudaMalloc(&cm->body_quat, m->nbody * 4 * sizeof(float));
    cudaMalloc(&cm->body_ipos, m->nbody * 3 * sizeof(float));
    cudaMalloc(&cm->body_iquat, m->nbody * 4 * sizeof(float));
    cudaMalloc(&cm->jnt_type, m->njnt * sizeof(int));
    cudaMalloc(&cm->jnt_qposadr, m->njnt * sizeof(int));
    cudaMalloc(&cm->jnt_axis, m->njnt * 3 * sizeof(float));
    cudaMalloc(&cm->jnt_pos, m->njnt * 3 * sizeof(float));
    cudaMalloc(&cm->geom_bodyid, m->ngeom * sizeof(int));
    cudaMalloc(&cm->geom_pos, m->ngeom * 3 * sizeof(float));
    cudaMalloc(&cm->geom_quat, m->ngeom * 4 * sizeof(float));
    cudaMalloc(&cm->site_bodyid, m->nsite * sizeof(int));
    cudaMalloc(&cm->site_pos, m->nsite * 3 * sizeof(float));
    cudaMalloc(&cm->site_quat, m->nsite * 4 * sizeof(float));

    cudaMemcpy(cm->qpos0, qpos0_float.data(), m->nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_jntadr, m->body_jntadr, m->njnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_jntnum, m->body_jntnum, m->njnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_parentid, m->body_parentid, m->nbody * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_mocapid, m->body_mocapid, m->nbody * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_pos, body_pos_float.data(), m->nbody * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_quat, body_quat_float.data(), m->nbody * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_ipos, body_ipos_float.data(), m->nbody * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_iquat, body_iquat_float.data(), m->nbody * 4 * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(cm->jnt_type, m->jnt_type, m->njnt * sizeof(int), cudaMemcpyHostToDevice);   
    cudaMemcpy(cm->jnt_qposadr, m->jnt_qposadr, m->njnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->jnt_axis, jnt_axis_float.data(), m->njnt * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->jnt_pos, jnt_pos_float.data(), m->njnt * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->geom_bodyid, m->geom_bodyid, m->ngeom * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->geom_pos, geom_pos_float.data(), m->ngeom * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->geom_quat, geom_quat_float.data(), m->ngeom * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->site_bodyid, m->site_bodyid, m->nsite * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->site_pos, site_pos_float.data(), m->nsite * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->site_quat, site_quat_float.data(), m->nsite * 4 * sizeof(float), cudaMemcpyHostToDevice);
    
    return cm;
}

CudaData* put_data(const mjModel* m, const mjData* d, int nworlds) {
    CudaData* cd = new CudaData();

    cd->nq = m->nq;
    cd->nmocap = m->nmocap;
    cd->nbody = m->nbody;
    cd->ngeom = m->ngeom;
    cd->nsite = m->nsite;
    cd->batch_size = nworlds;

    // Allocate temporary float arrays for conversion
    std::vector<float> qpos_float(nworlds * m->nq);
    std::vector<float> mocap_pos_float(nworlds * m->nmocap * 3);
    std::vector<float> mocap_quat_float(nworlds * m->nmocap * 4);
    std::vector<float> xanchor_float(nworlds * m->njnt * 3);
    std::vector<float> xaxis_float(nworlds * m->njnt * 3);
    std::vector<float> xmat_float(nworlds * m->nbody * 9);
    std::vector<float> xpos_float(nworlds * m->nbody * 3);
    std::vector<float> xquat_float(nworlds * m->nbody * 4);
    std::vector<float> xipos_float(nworlds * m->nbody * 3);
    std::vector<float> ximat_float(nworlds * m->nbody * 9);
    std::vector<float> geom_xpos_float(nworlds * m->ngeom * 3);
    std::vector<float> geom_xmat_float(nworlds * m->ngeom * 9);
    std::vector<float> site_xpos_float(nworlds * m->nsite * 3);
    std::vector<float> site_xmat_float(nworlds * m->nsite * 9);

    // Convert and tile data across nworlds
    for (int w = 0; w < nworlds; w++) {
        // qpos
        for (int i = 0; i < m->nq; i++) {
            qpos_float[w * m->nq + i] = static_cast<float>(d->qpos[i]);
        }
        
        // mocap data
        for (int i = 0; i < m->nmocap * 3; i++) {
            mocap_pos_float[w * m->nmocap * 3 + i] = static_cast<float>(d->mocap_pos[i]);
        }
        for (int i = 0; i < m->nmocap * 4; i++) {
            mocap_quat_float[w * m->nmocap * 4 + i] = static_cast<float>(d->mocap_quat[i]);
        }

        // joint data
        for (int i = 0; i < m->njnt * 3; i++) {
            xanchor_float[w * m->njnt * 3 + i] = static_cast<float>(d->xanchor[i]);
            xaxis_float[w * m->njnt * 3 + i] = static_cast<float>(d->xaxis[i]);
        }

        // body data
        for (int i = 0; i < m->nbody * 9; i++) {
            xmat_float[w * m->nbody * 9 + i] = static_cast<float>(d->xmat[i]);
            ximat_float[w * m->nbody * 9 + i] = static_cast<float>(d->ximat[i]);
        }
        for (int i = 0; i < m->nbody * 3; i++) {
            xpos_float[w * m->nbody * 3 + i] = static_cast<float>(d->xpos[i]);
            xipos_float[w * m->nbody * 3 + i] = static_cast<float>(d->xipos[i]);
        }
        for (int i = 0; i < m->nbody * 4; i++) {
            xquat_float[w * m->nbody * 4 + i] = static_cast<float>(d->xquat[i]);
        }

        // geom data
        for (int i = 0; i < m->ngeom * 3; i++) {
            geom_xpos_float[w * m->ngeom * 3 + i] = static_cast<float>(d->geom_xpos[i]);
        }
        for (int i = 0; i < m->ngeom * 9; i++) {
            geom_xmat_float[w * m->ngeom * 9 + i] = static_cast<float>(d->geom_xmat[i]);
        }

        // site data
        for (int i = 0; i < m->nsite * 3; i++) {
            site_xpos_float[w * m->nsite * 3 + i] = static_cast<float>(d->site_xpos[i]);
        }
        for (int i = 0; i < m->nsite * 9; i++) {
            site_xmat_float[w * m->nsite * 9 + i] = static_cast<float>(d->site_xmat[i]);
        }
    }

    cudaMalloc(&cd->qpos, nworlds * m->nq * sizeof(float));
    cudaMalloc(&cd->mocap_pos, nworlds * m->nmocap * 3 * sizeof(float));
    cudaMalloc(&cd->mocap_quat, nworlds * m->nmocap * 4 * sizeof(float));
    cudaMalloc(&cd->xanchor, nworlds * m->njnt * 3 * sizeof(float));
    cudaMalloc(&cd->xaxis, nworlds * m->njnt * 3 * sizeof(float));
    cudaMalloc(&cd->xmat, nworlds * m->nbody * 9 * sizeof(float));
    cudaMalloc(&cd->xpos, nworlds * m->nbody * 3 * sizeof(float));
    cudaMalloc(&cd->xquat, nworlds * m->nbody * 4 * sizeof(float));
    cudaMalloc(&cd->xipos, nworlds * m->nbody * 3 * sizeof(float));
    cudaMalloc(&cd->ximat, nworlds * m->nbody * 9 * sizeof(float));
    cudaMalloc(&cd->geom_xpos, nworlds * m->ngeom * 3 * sizeof(float));
    cudaMalloc(&cd->geom_xmat, nworlds * m->ngeom * 9 * sizeof(float));
    cudaMalloc(&cd->site_xpos, nworlds * m->nsite * 3 * sizeof(float));
    cudaMalloc(&cd->site_xmat, nworlds * m->nsite * 9 * sizeof(float));

    cudaMemcpy(cd->qpos, qpos_float.data(), nworlds * m->nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->mocap_pos, mocap_pos_float.data(), nworlds * m->nmocap * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->mocap_quat, mocap_quat_float.data(), nworlds * m->nmocap * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xanchor, xanchor_float.data(), nworlds * m->njnt * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xaxis, xaxis_float.data(), nworlds * m->njnt * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xmat, xmat_float.data(), nworlds * m->nbody * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xpos, xpos_float.data(), nworlds * m->nbody * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xquat, xquat_float.data(), nworlds * m->nbody * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xipos, xipos_float.data(), nworlds * m->nbody * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->ximat, ximat_float.data(), nworlds * m->nbody * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->geom_xpos, geom_xpos_float.data(), nworlds * m->ngeom * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->geom_xmat, geom_xmat_float.data(), nworlds * m->ngeom * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->site_xpos, site_xpos_float.data(), nworlds * m->nsite * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->site_xmat, site_xmat_float.data(), nworlds * m->nsite * 9 * sizeof(float), cudaMemcpyHostToDevice);

    return cd;
}

void free_cuda_model(CudaModel* cm) {
  if (cm) {
    delete cm;
  }
}

void free_cuda_data(CudaData* cd) {
  if (cd) {
    delete cd;
  }
} 