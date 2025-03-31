#include "smooth.h"

void LaunchNoiseKernel(
    unsigned int batch_size,
    CudaModel* cm,
    CudaData* cd) {

  int threadsPerBlock = 256;
  int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
  NoiseInjectionKernel<<<numBlocks, threadsPerBlock>>>(
      batch_size,
      cm->nq,
      cd->qpos,
      0.01f,
      0
  );
}

void LaunchKinematicsKernel(
    cudaStream_t stream,
    unsigned int batch_size,
    CudaModel* cm,
    CudaData* cd) {

  constexpr int threadsPerBlock = 256;
  int batchBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

  RootKernel<<<batchBlocks, threadsPerBlock, 0, stream>>>(
      batch_size,
      cd->xpos,
      cd->xquat,
      cd->xipos,
      cd->xmat,
      cd->ximat);

  for (int i = 1; i < cm->nbody_treeadr; i++) {
    int beg = cm->body_treeadr[i];
    int end = (i == cm->nbody_treeadr - 1) ? cm->nbody : cm->body_treeadr[i + 1];
    int numBodiesInLevel = end - beg;

    dim3 gridDim(batchBlocks, numBodiesInLevel);
    
    LevelKernel<<<gridDim, threadsPerBlock, 0, stream>>>(
        batch_size,
        beg,
        cm->nq,
        cm->njnt,
        cm->nbody,
        cm->qpos0,
        cm->body_jntadr,
        cm->body_jntnum,
        cm->body_parentid,
        cm->body_mocapid,
        cm->body_pos,
        cm->body_quat,
        cm->body_ipos,
        cm->body_iquat,
        cm->jnt_type,
        cm->jnt_qposadr,
        cm->jnt_axis,
        cm->jnt_pos,
        cm->body_tree,
        cd->qpos,
        cd->mocap_pos,
        cd->mocap_quat,
        cd->xanchor,
        cd->xaxis,
        cd->xmat,
        cd->xpos,
        cd->xquat,
        cd->xipos,
        cd->ximat);
  }

  if (cm->ngeom > 0) {
    dim3 gridDim(batchBlocks, cm->ngeom);
    GeomLocalToGlobalKernel<<<gridDim, threadsPerBlock, 0, stream>>>(
        batch_size,
        cm->nbody,
        cm->ngeom,
        cm->geom_bodyid,
        cm->geom_pos,
        cm->geom_quat,
        cd->xpos,
        cd->xquat,
        cd->geom_xpos,
        cd->geom_xmat);
  }

  if (cm->nsite > 0) {
    dim3 gridDim(batchBlocks, cm->nsite);
    SiteLocalToGlobalKernel<<<gridDim, threadsPerBlock, 0, stream>>>(
        batch_size,
        cm->nbody,
        cm->nsite,
        cm->site_bodyid,
        cm->site_pos,
        cm->site_quat,
        cd->xpos,
        cd->xquat,
        cd->site_xpos,
        cd->site_xmat);
  }
}

__global__ void RootKernel(
    unsigned int n,
    vec3p* xpos,
    quat* xquat,
    vec3p* xipos,
    float* xmat,
    float* ximat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n) {
    return;
  }

  xpos[tid] = {0.0f, 0.0f, 0.0f, 0.0f};
  xipos[tid] = {0.0f, 0.0f, 0.0f, 0.0f};
  
  xquat[tid] = {1.0f, 0.0f, 0.0f, 0.0f};
  
  xmat[tid * 9] = 1.0f;
  xmat[tid * 9 + 1] = 0.0f;
  xmat[tid * 9 + 2] = 0.0f;
  xmat[tid * 9 + 3] = 0.0f;
  xmat[tid * 9 + 4] = 1.0f;
  xmat[tid * 9 + 5] = 0.0f;
  xmat[tid * 9 + 6] = 0.0f;
  xmat[tid * 9 + 7] = 0.0f;
  xmat[tid * 9 + 8] = 1.0f;
  
  ximat[tid * 9] = 1.0f;
  ximat[tid * 9 + 1] = 0.0f;
  ximat[tid * 9 + 2] = 0.0f;
  ximat[tid * 9 + 3] = 0.0f;
  ximat[tid * 9 + 4] = 1.0f;
  ximat[tid * 9 + 5] = 0.0f;
  ximat[tid * 9 + 6] = 0.0f;
  ximat[tid * 9 + 7] = 0.0f;
  ximat[tid * 9 + 8] = 1.0f;
}

__global__ void LevelKernel(
    unsigned int n,
    unsigned int leveladr,
    unsigned int nq,
    unsigned int njnt,
    unsigned int nbody,
    const float* qpos0,
    const int* body_jntadr,
    const int* body_jntnum,
    const int* body_parentid,
    const int* body_mocapid,
    const vec3p* body_pos,
    const quat* body_quat,
    const vec3p* body_ipos,
    const quat* body_iquat,
    const int* jnt_type,
    const int* jnt_qposadr,
    const vec3p* jnt_axis,
    const vec3p* jnt_pos,
    const int* body_tree,
    float* qpos,
    vec3p* mocap_pos,
    quat* mocap_quat,
    vec3p* xanchor,
    vec3p* xaxis,
    float* xmat,
    vec3p* xpos,
    quat* xquat,
    vec3p* xipos,
    float* ximat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int nodeid = blockIdx.y;

  if (tid >= n) return;
  const int bodyid = body_tree[leveladr + nodeid];
  const int jntadr = body_jntadr[bodyid];
  const int jntnum = body_jntnum[bodyid];
  const unsigned int qpos_offset = tid * nq;
  const unsigned int body_offset = tid * nbody;
  const unsigned int jnt_offset = tid * njnt;

  vec3p lxpos = {0.0f, 0.0f, 0.0f, 0.0f};
  quat lxquat = {0.0f, 0.0f, 0.0f, 0.0f};

  if (jntnum == 0) {
    int pid = body_parentid[bodyid];
    MulMatVec3(lxpos, xmat + (body_offset * 9 + pid * 9), body_pos[bodyid]);
    lxpos += xpos[body_offset + pid];
    
    MulQuat(lxquat, xquat[body_offset + pid], body_quat[bodyid]);
  } 
  else if (jntnum == 1 && jnt_type[jntadr] == mjJNT_FREE) {
    // Free joint (direct set from qpos)
    int qadr = jnt_qposadr[jntadr];
    float* src_qpos = qpos + qpos_offset + qadr;
    
    lxpos.x = src_qpos[0];
    lxpos.y = src_qpos[1];
    lxpos.z = src_qpos[2];
    lxquat.w = src_qpos[3];
    lxquat.x = src_qpos[4];
    lxquat.y = src_qpos[5];
    lxquat.z = src_qpos[6];

    xanchor[jnt_offset + jntadr] = lxpos;
    xaxis[jnt_offset + jntadr] = jnt_axis[jntadr];
  } 
  else {
    int pid = body_parentid[bodyid];
    MulMatVec3(lxpos, xmat + (body_offset * 9 + pid * 9), body_pos[bodyid]);
    lxpos += xpos[body_offset + pid];
    
    MulQuat(lxquat, xquat[body_offset + pid], body_quat[bodyid]);

    float* curr_qpos = qpos + qpos_offset;
    
    for (int j = 0; j < jntnum; j++) {
      int jid = jntadr + j;
      int qadr = jnt_qposadr[jid];
      int jtype = jnt_type[jid];
      
      vec3p vec = {0.0f, 0.0f, 0.0f, 0.0f};
      RotVecQuat(vec, jnt_pos[jid], lxquat);
      
      xanchor[jnt_offset + jid] = vec + lxpos;

      RotVecQuat(xaxis[jnt_offset + jid], jnt_axis[jid], lxquat);

      switch (jtype) {
        case mjJNT_SLIDE: {
          float qpos_diff = curr_qpos[qadr] - qpos0[qadr];
          lxpos += xaxis[jnt_offset + jid] * qpos_diff;
          break;
        }

        case mjJNT_BALL: {
          quat qloc = {0.0f, 0.0f, 0.0f, 0.0f};
          qloc.w = curr_qpos[qadr];
          qloc.x = curr_qpos[qadr+1];
          qloc.y = curr_qpos[qadr+2];
          qloc.z = curr_qpos[qadr+3];
          
          MulQuat(lxquat, lxquat, qloc);
          
          RotVecQuat(vec, jnt_pos[jid], lxquat);
          lxpos = xanchor[jnt_offset + jid] - vec;
          break;
        }

        case mjJNT_HINGE: {
          float angle = curr_qpos[qadr] - qpos0[qadr];
          quat qloc = {0.0f, 0.0f, 0.0f, 0.0f};
          AxisAngle2Quat(qloc, jnt_axis[jid], angle);
          
          MulQuat(lxquat, lxquat, qloc);
          
          RotVecQuat(vec, jnt_pos[jid], lxquat);
          lxpos = xanchor[jnt_offset + jid] - vec;
          break;
        }
      }
    }
  }

  float norm = sqrtf(lxquat.w*lxquat.w + lxquat.x*lxquat.x + 
                     lxquat.y*lxquat.y + lxquat.z*lxquat.z);
  float invNorm = 1.0f / norm;
  lxquat.w *= invNorm;
  lxquat.x *= invNorm;
  lxquat.y *= invNorm;
  lxquat.z *= invNorm;

  xquat[body_offset + bodyid] = lxquat;
  
  xpos[body_offset + bodyid] = lxpos;
  
  Quat2Mat(xmat + (body_offset * 9 + bodyid * 9), lxquat);

  vec3p vec = {0.0f, 0.0f, 0.0f, 0.0f};
  RotVecQuat(vec, body_ipos[bodyid], lxquat);
  xipos[body_offset + bodyid] = vec + lxpos;

  quat quat = {0.0f, 0.0f, 0.0f, 0.0f};
  MulQuat(quat, lxquat, body_iquat[bodyid]);
  Quat2Mat(ximat + (body_offset * 9 + bodyid * 9), quat);
}

__global__ void GeomLocalToGlobalKernel(
    unsigned int n,
    unsigned int nbody,
    unsigned int ngeom,
    const int* geom_bodyid,
    const vec3p* geom_pos,
    const quat* geom_quat,
    const vec3p* xpos,
    const quat* xquat,
    vec3p* geom_xpos,
    float* geom_xmat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int geomid = blockIdx.y;

  if (tid >= n) return;

  const int bodyid = geom_bodyid[geomid];
  const unsigned int body_offset = tid * nbody;
  const unsigned int geom_offset = tid * ngeom;

  vec3p vec = {0.0f, 0.0f, 0.0f, 0.0f};
  RotVecQuat(vec, geom_pos[geomid], xquat[bodyid]);
  geom_xpos[geom_offset + geomid] = vec + xpos[body_offset + bodyid];

  quat quat = {0.0f, 0.0f, 0.0f, 0.0f};
  MulQuat(quat, xquat[bodyid], geom_quat[geomid]);
  Quat2Mat(geom_xmat + geom_offset * 9 + geomid * 9, quat);
}

__global__ void SiteLocalToGlobalKernel(
    unsigned int n,
    unsigned int nbody,
    unsigned int nsite,
    const int* site_bodyid,
    const vec3p* site_pos,
    const quat* site_quat,
    const vec3p* xpos,
    const quat* xquat,
    vec3p* site_xpos,
    float* site_xmat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int siteid = blockIdx.y;

  if (tid >= n) return;

  const int bodyid = site_bodyid[siteid];
  const unsigned int body_offset = tid * nbody;
  const unsigned int site_offset = tid * nsite;

  vec3p vec = {0.0f, 0.0f, 0.0f, 0.0f};
  RotVecQuat(vec, site_pos[siteid], xquat[bodyid]);
  site_xpos[site_offset + siteid] = vec + xpos[body_offset + bodyid];

  quat quat = {0.0f, 0.0f, 0.0f, 0.0f};
  MulQuat(quat, xquat[bodyid], site_quat[siteid]);
  Quat2Mat(site_xmat + site_offset * 9 + siteid * 9, quat);
}

// Noise injection kernel
__global__ void NoiseInjectionKernel(
    unsigned int n,
    unsigned int nq,
    float* qpos,
    float noise_scale,
    unsigned int seed
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float* batch_qpos = qpos + tid * nq;
    for (int i = 0; i < nq; i++) {
        batch_qpos[i] += 0.001f;
    }
}