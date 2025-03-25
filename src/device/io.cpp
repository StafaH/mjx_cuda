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

    // Compute body tree and level addresses
    std::vector<int> body_depth(m->nbody, -1);
    std::vector<std::vector<int>> bodies_by_level;
    body_depth[0] = 0;  // Root body is at level 0
    
    // Compute depths and group bodies by level
    for (int i = 1; i < m->nbody; i++) {
        body_depth[i] = body_depth[m->body_parentid[i]] + 1;
    }
    
    // Find max depth
    int max_depth = *std::max_element(body_depth.begin(), body_depth.end());
    
    // Group bodies by level
    bodies_by_level.resize(max_depth + 1);
    for (int i = 0; i < m->nbody; i++) {
        bodies_by_level[body_depth[i]].push_back(i);
    }
    
    // Create body_tree array (BFS ordering)
    std::vector<int> body_tree;
    for (const auto& level_bodies : bodies_by_level) {
        body_tree.insert(body_tree.end(), level_bodies.begin(), level_bodies.end());
    }
    
    // Create body_leveladr array (starting indices for each level)
    std::vector<int> body_leveladr;
    int offset = 0;
    for (const auto& level_bodies : bodies_by_level) {
        body_leveladr.push_back(offset);
        offset += level_bodies.size();
    }
    
    // Allocate GPU memory with correct sizes
    cudaMalloc(&cm->body_tree, m->nbody * sizeof(int));
    cudaMalloc(&cm->body_leveladr, (max_depth + 1) * sizeof(int));
    
    // Copy arrays to GPU
    cudaMemcpy(cm->body_tree, body_tree.data(), m->nbody * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_leveladr, body_leveladr.data(), (max_depth + 1) * sizeof(int), cudaMemcpyHostToDevice);

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

    cudaMemcpy(cm->qpos0, m->qpos0, m->nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_jntadr, m->body_jntadr, m->njnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_jntnum, m->body_jntnum, m->njnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_parentid, m->body_parentid, m->nbody * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_mocapid, m->body_mocapid, m->nbody * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_pos, m->body_pos, m->nbody * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_quat, m->body_quat, m->nbody * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_ipos, m->body_ipos, m->nbody * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_iquat, m->body_iquat, m->nbody * 4 * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(cm->jnt_type, m->jnt_type, m->njnt * sizeof(int), cudaMemcpyHostToDevice);   
    cudaMemcpy(cm->jnt_qposadr, m->jnt_qposadr, m->njnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->jnt_axis, m->jnt_axis, m->njnt * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->jnt_pos, m->jnt_pos, m->njnt * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->geom_bodyid, m->geom_bodyid, m->ngeom * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->geom_pos, m->geom_pos, m->ngeom * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->geom_quat, m->geom_quat, m->ngeom * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->site_bodyid, m->site_bodyid, m->nsite * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->site_pos, m->site_pos, m->nsite * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->site_quat, m->site_quat, m->nsite * 4 * sizeof(float), cudaMemcpyHostToDevice);
    
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

    cudaMemcpy(cd->qpos, d->qpos, nworlds * m->nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->mocap_pos, d->mocap_pos, nworlds * m->nmocap * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->mocap_quat, d->mocap_quat, nworlds * m->nmocap * 4 * sizeof(float), cudaMemcpyHostToDevice);

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