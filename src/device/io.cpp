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
    std::vector<vec3p> body_pos(m->nbody);
    std::vector<quat> body_quat(m->nbody);
    std::vector<vec3p> body_ipos(m->nbody);
    std::vector<quat> body_iquat(m->nbody);
    std::vector<vec3p> jnt_axis(m->njnt);
    std::vector<vec3p> jnt_pos(m->njnt);
    std::vector<vec3p> geom_pos(m->ngeom);
    std::vector<quat> geom_quat(m->ngeom);
    std::vector<vec3p> site_pos(m->nsite);
    std::vector<quat> site_quat(m->nsite);

    // Convert double arrays to float
    for (int i = 0; i < m->nq; i++) {
        qpos0_float[i] = static_cast<float>(m->qpos0[i]);
    }

    for (int i = 0; i < m->nbody; i++) {
        vec3p bpos = {
            static_cast<float>(m->body_pos[i * 3]),
            static_cast<float>(m->body_pos[i * 3 + 1]),
            static_cast<float>(m->body_pos[i * 3 + 2]),
            0.0f
        };
        body_pos[i] = bpos;

        vec3p bposi = {
            static_cast<float>(m->body_ipos[i * 3]),
            static_cast<float>(m->body_ipos[i * 3 + 1]),
            static_cast<float>(m->body_ipos[i * 3 + 2]),
            0.0f
        };
        body_ipos[i] = bposi;
    }

    for (int i = 0; i < m->nbody; i++) {
        quat bquat = {
            static_cast<float>(m->body_quat[i * 4]),
            static_cast<float>(m->body_quat[i * 4 + 1]),
            static_cast<float>(m->body_quat[i * 4 + 2]),
            static_cast<float>(m->body_quat[i * 4 + 3])
        };
        body_quat[i] = bquat;

        quat biquat = {
            static_cast<float>(m->body_iquat[i * 4]),
            static_cast<float>(m->body_iquat[i * 4 + 1]),
            static_cast<float>(m->body_iquat[i * 4 + 2]),
            static_cast<float>(m->body_iquat[i * 4 + 3])
        };
        body_iquat[i] = biquat;
    }
    
    for (int i = 0; i < m->njnt; i++) {
        vec3p jaxis = {
            static_cast<float>(m->jnt_axis[i * 3]),
            static_cast<float>(m->jnt_axis[i * 3 + 1]),
            static_cast<float>(m->jnt_axis[i * 3 + 2]),
            0.0f
        };
        jnt_axis[i] = jaxis;

        vec3p jpos = {
            static_cast<float>(m->jnt_pos[i * 3]),
            static_cast<float>(m->jnt_pos[i * 3 + 1]),
            static_cast<float>(m->jnt_pos[i * 3 + 2]),
            0.0f
        };
        jnt_pos[i] = jpos;
    }

    for (int i = 0; i < m->ngeom; i++) {
        vec3p gpos = {
            static_cast<float>(m->geom_pos[i * 3]),
            static_cast<float>(m->geom_pos[i * 3 + 1]),
            static_cast<float>(m->geom_pos[i * 3 + 2]),
            0.0f
        };
        geom_pos[i] = gpos;

        quat gquat = {
            static_cast<float>(m->geom_quat[i * 4]),
            static_cast<float>(m->geom_quat[i * 4 + 1]),
            static_cast<float>(m->geom_quat[i * 4 + 2]),
            static_cast<float>(m->geom_quat[i * 4 + 3])
        };
        geom_quat[i] = gquat;
    }

    for (int i = 0; i < m->nsite; i++) {
        vec3p spos = {
            static_cast<float>(m->site_pos[i * 3]),
            static_cast<float>(m->site_pos[i * 3 + 1]),
            static_cast<float>(m->site_pos[i * 3 + 2]),
            0.0f
        };
        site_pos[i] = spos;

        quat squat = {
            static_cast<float>(m->site_quat[i * 4]),
            static_cast<float>(m->site_quat[i * 4 + 1]),
            static_cast<float>(m->site_quat[i * 4 + 2]),
            static_cast<float>(m->site_quat[i * 4 + 3])
        };
        site_quat[i] = squat;
    }

    cudaMalloc(&cm->qpos0, m->nq * sizeof(float));
    cudaMalloc(&cm->body_jntadr, m->njnt * sizeof(int));
    cudaMalloc(&cm->body_jntnum, m->njnt * sizeof(int));
    cudaMalloc(&cm->body_parentid, m->nbody * sizeof(int));
    cudaMalloc(&cm->body_mocapid, m->nbody * sizeof(int));
    cudaMalloc(&cm->body_pos, m->nbody * sizeof(vec3p));
    cudaMalloc(&cm->body_quat, m->nbody * sizeof(quat));
    cudaMalloc(&cm->body_ipos, m->nbody * sizeof(vec3p));
    cudaMalloc(&cm->body_iquat, m->nbody * sizeof(quat));
    cudaMalloc(&cm->jnt_type, m->njnt * sizeof(int));
    cudaMalloc(&cm->jnt_qposadr, m->njnt * sizeof(int));
    cudaMalloc(&cm->jnt_axis, m->njnt * sizeof(vec3p));
    cudaMalloc(&cm->jnt_pos, m->njnt * sizeof(vec3p));
    cudaMalloc(&cm->geom_bodyid, m->ngeom * sizeof(int));
    cudaMalloc(&cm->geom_pos, m->ngeom * sizeof(vec3p));
    cudaMalloc(&cm->geom_quat, m->ngeom * sizeof(quat));
    cudaMalloc(&cm->site_bodyid, m->nsite * sizeof(int));
    cudaMalloc(&cm->site_pos, m->nsite * sizeof(vec3p));
    cudaMalloc(&cm->site_quat, m->nsite * sizeof(quat));

    cudaMemcpy(cm->qpos0, qpos0_float.data(), m->nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_jntadr, m->body_jntadr, m->njnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_jntnum, m->body_jntnum, m->njnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_parentid, m->body_parentid, m->nbody * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_mocapid, m->body_mocapid, m->nbody * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_pos, body_pos.data(), m->nbody * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_quat, body_quat.data(), m->nbody * sizeof(quat), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_ipos, body_ipos.data(), m->nbody * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->body_iquat, body_iquat.data(), m->nbody * sizeof(quat), cudaMemcpyHostToDevice);    
    cudaMemcpy(cm->jnt_type, m->jnt_type, m->njnt * sizeof(int), cudaMemcpyHostToDevice);   
    cudaMemcpy(cm->jnt_qposadr, m->jnt_qposadr, m->njnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->jnt_axis, jnt_axis.data(), m->njnt * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->jnt_pos, jnt_pos.data(), m->njnt * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->geom_bodyid, m->geom_bodyid, m->ngeom * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->geom_pos, geom_pos.data(), m->ngeom * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->geom_quat, geom_quat.data(), m->ngeom * sizeof(quat), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->site_bodyid, m->site_bodyid, m->nsite * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->site_pos, site_pos.data(), m->nsite * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cm->site_quat, site_quat.data(), m->nsite * sizeof(quat), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(cu_qpos0, qpos0_float.data(), m->nq * sizeof(float));
    cudaMemcpyToSymbol(cu_body_jntadr, m->body_jntadr, cm->njnt * sizeof(int));
    cudaMemcpyToSymbol(cu_body_jntnum, m->body_jntnum, cm->njnt * sizeof(int));
    cudaMemcpyToSymbol(cu_body_parentid, m->body_parentid, cm->nbody * sizeof(int));
    cudaMemcpyToSymbol(cu_body_mocapid, m->body_mocapid, cm->nbody * sizeof(int));
    cudaMemcpyToSymbol(cu_body_pos, body_pos.data(), cm->nbody * sizeof(vec3p));
    cudaMemcpyToSymbol(cu_body_quat, body_quat.data(), cm->nbody * sizeof(quat));
    cudaMemcpyToSymbol(cu_body_ipos, body_ipos.data(), cm->nbody * sizeof(vec3p));
    cudaMemcpyToSymbol(cu_body_iquat, body_iquat.data(), cm->nbody * sizeof(quat));
    cudaMemcpyToSymbol(cu_jnt_type, m->jnt_type, cm->njnt * sizeof(int));
    cudaMemcpyToSymbol(cu_jnt_qposadr, m->jnt_qposadr, cm->njnt * sizeof(int));
    cudaMemcpyToSymbol(cu_jnt_axis, jnt_axis.data(), cm->njnt * sizeof(vec3p));
    cudaMemcpyToSymbol(cu_jnt_pos, jnt_pos.data(), cm->njnt * sizeof(vec3p));
    cudaDeviceSynchronize();
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
    std::vector<float> qpos(nworlds * m->nq);
    std::vector<vec3p> mocap_pos(nworlds * m->nmocap);
    std::vector<quat> mocap_quat(nworlds * m->nmocap);
    std::vector<vec3p> xanchor(nworlds * m->njnt);
    std::vector<vec3p> xaxis(nworlds * m->njnt);
    std::vector<mat3p> xmat(nworlds * m->nbody);
    std::vector<vec3p> xpos(nworlds * m->nbody);
    std::vector<quat> xquat(nworlds * m->nbody);
    std::vector<vec3p> xipos(nworlds * m->nbody);
    std::vector<mat3p> ximat(nworlds * m->nbody);
    std::vector<vec3p> geom_xpos(nworlds * m->ngeom);
    std::vector<mat3p> geom_xmat(nworlds * m->ngeom);
    std::vector<vec3p> site_xpos(nworlds * m->nsite);
    std::vector<mat3p> site_xmat(nworlds * m->nsite);

    // Convert and tile data across nworlds
    for (int w = 0; w < nworlds; w++) {
        for (int i = 0; i < m->nq; i++) {
            qpos[w * m->nq + i] = static_cast<float>(d->qpos[i]);
        }
        
        for (int i = 0; i < m->nmocap; i++) {
            vec3p mpos = {
                static_cast<float>(d->mocap_pos[i * 3]),
                static_cast<float>(d->mocap_pos[i * 3 + 1]),
                static_cast<float>(d->mocap_pos[i * 3 + 2]),
                0.0f
            };
            mocap_pos[w * m->nmocap + i] = mpos;
        }
        for (int i = 0; i < m->nmocap; i++) {
            quat mquat = {
                static_cast<float>(d->mocap_quat[i * 4]),
                static_cast<float>(d->mocap_quat[i * 4 + 1]),
                static_cast<float>(d->mocap_quat[i * 4 + 2]),
                static_cast<float>(d->mocap_quat[i * 4 + 3])
            };
            mocap_quat[w * m->nmocap + i] = mquat;
        }

        // joint data
        for (int i = 0; i < m->njnt; i++) {
            vec3p anchor = {
                static_cast<float>(d->xanchor[i * 3]),
                static_cast<float>(d->xanchor[i * 3 + 1]),
                static_cast<float>(d->xanchor[i * 3 + 2]),
                0.0f
            };
            xanchor[w * m->njnt + i] = anchor;
            vec3p axis = {
                static_cast<float>(d->xaxis[i * 3]),
                static_cast<float>(d->xaxis[i * 3 + 1]),
                static_cast<float>(d->xaxis[i * 3 + 2]),
                0.0f
            };
            xaxis[w * m->njnt + i] = axis;
        }

        // body data
        for (int i = 0; i < m->nbody; i++) {
            mat3p xm = {
                static_cast<float>(d->xmat[i * 9]),
                static_cast<float>(d->xmat[i * 9 + 1]),
                static_cast<float>(d->xmat[i * 9 + 2]),
                static_cast<float>(d->xmat[i * 9 + 3]),
                static_cast<float>(d->xmat[i * 9 + 4]),
                static_cast<float>(d->xmat[i * 9 + 5]),
                static_cast<float>(d->xmat[i * 9 + 6]),
                static_cast<float>(d->xmat[i * 9 + 7]),
                static_cast<float>(d->xmat[i * 9 + 8])
            };
            xmat[w * m->nbody + i] = xm;

            mat3p xim = {
                static_cast<float>(d->ximat[i * 9]),
                static_cast<float>(d->ximat[i * 9 + 1]),
                static_cast<float>(d->ximat[i * 9 + 2]),
                static_cast<float>(d->ximat[i * 9 + 3]),
                static_cast<float>(d->ximat[i * 9 + 4]),
                static_cast<float>(d->ximat[i * 9 + 5]),
                static_cast<float>(d->ximat[i * 9 + 6]),
                static_cast<float>(d->ximat[i * 9 + 7]),
                static_cast<float>(d->ximat[i * 9 + 8])
            };
            ximat[w * m->nbody + i] = xim;
        }

        for (int i = 0; i < m->nbody; i++) {
            vec3p xp = {
                static_cast<float>(d->xpos[i * 3]),
                static_cast<float>(d->xpos[i * 3 + 1]),
                static_cast<float>(d->xpos[i * 3 + 2]),
                0.0f
            };
            xpos[w * m->nbody + i] = xp;
            vec3p xip = {
                static_cast<float>(d->xipos[i * 3]),
                static_cast<float>(d->xipos[i * 3 + 1]),
                static_cast<float>(d->xipos[i * 3 + 2]),
                0.0f
            };
            xipos[w * m->nbody + i] = xip;
        }
        for (int i = 0; i < m->nbody; i++) {
            quat xq = {
                static_cast<float>(d->xquat[i * 4]),
                static_cast<float>(d->xquat[i * 4 + 1]),
                static_cast<float>(d->xquat[i * 4 + 2]),
                static_cast<float>(d->xquat[i * 4 + 3])
            };
            xquat[w * m->nbody + i] = xq;
        }

        for (int i = 0; i < m->ngeom; i++) {
            vec3p gpos = {
                static_cast<float>(d->geom_xpos[i * 3]),
                static_cast<float>(d->geom_xpos[i * 3 + 1]),
                static_cast<float>(d->geom_xpos[i * 3 + 2]),
                0.0f
            };
            geom_xpos[w * m->ngeom + i] = gpos;
        }
        for (int i = 0; i < m->ngeom; i++) {
            mat3p gmat = {
                static_cast<float>(d->geom_xmat[i * 9]),
                static_cast<float>(d->geom_xmat[i * 9 + 1]),
                static_cast<float>(d->geom_xmat[i * 9 + 2]),
                static_cast<float>(d->geom_xmat[i * 9 + 3]),
                static_cast<float>(d->geom_xmat[i * 9 + 4]),
                static_cast<float>(d->geom_xmat[i * 9 + 5]),
                static_cast<float>(d->geom_xmat[i * 9 + 6]),
                static_cast<float>(d->geom_xmat[i * 9 + 7]),
                static_cast<float>(d->geom_xmat[i * 9 + 8])
            };
            geom_xmat[w * m->ngeom + i] = gmat;
        }

        for (int i = 0; i < m->nsite; i++) {
            vec3p spos = {
                static_cast<float>(d->site_xpos[i * 3]),
                static_cast<float>(d->site_xpos[i * 3 + 1]),
                static_cast<float>(d->site_xpos[i * 3 + 2]),
                0.0f
            };
            site_xpos[w * m->nsite + i] = spos;
        }
        for (int i = 0; i < m->nsite; i++) {
            mat3p smat = {
                static_cast<float>(d->site_xmat[i * 9]),
                static_cast<float>(d->site_xmat[i * 9 + 1]),
                static_cast<float>(d->site_xmat[i * 9 + 2]),
                static_cast<float>(d->site_xmat[i * 9 + 3]),
                static_cast<float>(d->site_xmat[i * 9 + 4]),
                static_cast<float>(d->site_xmat[i * 9 + 5]),
                static_cast<float>(d->site_xmat[i * 9 + 6]),
                static_cast<float>(d->site_xmat[i * 9 + 7]),
                static_cast<float>(d->site_xmat[i * 9 + 8])
            };
            site_xmat[w * m->nsite + i] = smat;
        }
    }

    cudaMalloc(&cd->qpos, nworlds * m->nq * sizeof(float));
    cudaMalloc(&cd->mocap_pos, nworlds * m->nmocap * sizeof(vec3p));
    cudaMalloc(&cd->mocap_quat, nworlds * m->nmocap * sizeof(quat));
    cudaMalloc(&cd->xanchor, nworlds * m->njnt * sizeof(vec3p));
    cudaMalloc(&cd->xaxis, nworlds * m->njnt * sizeof(vec3p));
    cudaMalloc(&cd->xmat, nworlds * m->nbody * sizeof(mat3p));
    cudaMalloc(&cd->xpos, nworlds * m->nbody * sizeof(vec3p));
    cudaMalloc(&cd->xquat, nworlds * m->nbody * sizeof(quat));
    cudaMalloc(&cd->xipos, nworlds * m->nbody * sizeof(vec3p));
    cudaMalloc(&cd->ximat, nworlds * m->nbody * sizeof(mat3p));
    cudaMalloc(&cd->geom_xpos, nworlds * m->ngeom * sizeof(vec3p));
    cudaMalloc(&cd->geom_xmat, nworlds * m->ngeom * sizeof(mat3p));
    cudaMalloc(&cd->site_xpos, nworlds * m->nsite * sizeof(vec3p));
    cudaMalloc(&cd->site_xmat, nworlds * m->nsite * sizeof(mat3p));

    cudaMemcpy(cd->qpos, qpos.data(), nworlds * m->nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->mocap_pos, mocap_pos.data(), nworlds * m->nmocap * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->mocap_quat, mocap_quat.data(), nworlds * m->nmocap * sizeof(quat), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xanchor, xanchor.data(), nworlds * m->njnt * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xaxis, xaxis.data(), nworlds * m->njnt * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xmat, xmat.data(), nworlds * m->nbody * sizeof(mat3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xpos, xpos.data(), nworlds * m->nbody * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xquat, xquat.data(), nworlds * m->nbody * sizeof(quat), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->xipos, xipos.data(), nworlds * m->nbody * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->ximat, ximat.data(), nworlds * m->nbody * sizeof(mat3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->geom_xpos, geom_xpos.data(), nworlds * m->ngeom * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->geom_xmat, geom_xmat.data(), nworlds * m->ngeom * sizeof(mat3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->site_xpos, site_xpos.data(), nworlds * m->nsite * sizeof(vec3p), cudaMemcpyHostToDevice);
    cudaMemcpy(cd->site_xmat, site_xmat.data(), nworlds * m->nsite * sizeof(mat3p), cudaMemcpyHostToDevice);
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