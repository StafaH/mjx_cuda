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
    mat3p* xmat,
    mat3p* ximat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n) {
    return;
  }

  xpos[tid] = {0.0f, 0.0f, 0.0f, 0.0f};
  xipos[tid] = {0.0f, 0.0f, 0.0f, 0.0f};
  
  xquat[tid] = {1.0f, 0.0f, 0.0f, 0.0f};
  
  xmat[tid] = {1.0f, 0.0f, 0.0f,
               0.0f, 1.0f, 0.0f,
               0.0f, 0.0f, 1.0f};

  ximat[tid] = {1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 1.0f};
}

__global__ void LevelKernel(
    const unsigned int n,
    const unsigned int leveladr,
    const unsigned int nq,
    const unsigned int njnt,
    const unsigned int nbody,
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
    mat3p* xmat,
    vec3p* xpos,
    quat* xquat,
    vec3p* xipos,
    mat3p* ximat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int nodeid = blockIdx.y;

  if (tid >= n) return;
  int bodyid = __ldg(&body_tree[leveladr + nodeid]);
  int jntadr = __ldg(&body_jntadr[bodyid]);
  int jntnum = __ldg(&body_jntnum[bodyid]);
  int qpos_offset = tid * nq;
  int body_offset = tid * nbody;
  int jnt_offset = tid * njnt;

  vec3p pos = {0.0f, 0.0f, 0.0f, 0.0f};
  quat rot = {0.0f, 0.0f, 0.0f, 0.0f};

  if (jntnum == 0) {
    int pid = __ldg(&body_parentid[bodyid]);
    pos = MulMatVec3(xmat[body_offset + pid], body_pos[bodyid]);
    pos += xpos[body_offset + pid];
    rot = MulQuat(xquat[body_offset + pid], body_quat[bodyid]);
  } 
  else if (jntnum == 1 && jnt_type[jntadr] == mjJNT_FREE) {
    float* src = qpos + qpos_offset + __ldg(&jnt_qposadr[jntadr]);
    pos = make_vec3p(src[0], src[1], src[2]);
    rot = {src[3], src[4], src[5], src[6]};

    xanchor[jnt_offset + jntadr] = pos;
    xaxis[jnt_offset + jntadr] = jnt_axis[jntadr];
  } 
  else {
    int pid = __ldg(&body_parentid[bodyid]);
    pos = MulMatVec3(xmat[body_offset + pid], body_pos[bodyid]);
    pos += xpos[body_offset + pid];
    
    rot = MulQuat(xquat[body_offset + pid], body_quat[bodyid]);
    
    for (int j = 0; j < jntnum; j++) {
      int jid = jntadr + j;
      int qadr = jnt_qposadr[jid];
      int jtype = jnt_type[jid];
      
      vec3p anchor = RotVecQuat(jnt_pos[jid], rot) + pos;
      xanchor[jnt_offset + jid] = anchor;
      xaxis[jnt_offset + jid] = RotVecQuat(jnt_axis[jid], rot);

      switch (jtype) {
        case mjJNT_SLIDE: {
          pos += xaxis[jnt_offset + jid] * (qpos[qpos_offset + qadr] - qpos0[qadr]);
          break;
        }

        case mjJNT_BALL: {
          quat qloc = {qpos[qpos_offset + qadr],
                       qpos[qpos_offset + qadr + 1],
                       qpos[qpos_offset + qadr + 2],
                       qpos[qpos_offset + qadr + 3]};
          rot = MulQuat(rot, qloc);
          pos = anchor - RotVecQuat(jnt_pos[jid], rot);
          break;
        }

        case mjJNT_HINGE: {
          float angle = qpos[qpos_offset + qadr] - qpos0[qadr];
          float s, c;
          __sincosf(0.5f * angle, &s, &c);
          quat qloc = {c,
                       jnt_axis[jid].x * s,
                       jnt_axis[jid].y * s,
                       jnt_axis[jid].z * s};
          rot = MulQuat(rot, qloc);
          pos = anchor - RotVecQuat(jnt_pos[jid], rot);
          break;
        }
      }
    }
  }

  NormalizeQuat(rot);
  xpos[body_offset + bodyid] = pos;
  xquat[body_offset + bodyid] = rot;

  Quat2Mat(xmat[body_offset + bodyid], rot);

  vec3p local_ipos = RotVecQuat(body_ipos[bodyid], rot);
  xipos[body_offset + bodyid] = local_ipos + pos;

  quat temp_iquat = MulQuat(rot, body_iquat[bodyid]);
  Quat2Mat(ximat[body_offset + bodyid], temp_iquat);
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
    mat3p* geom_xmat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int geomid = blockIdx.y;
  if (tid >= n) return;

  const int bodyid = __ldg(&geom_bodyid[geomid]);
  const unsigned int body_offset = tid * nbody + bodyid;
  const unsigned int geom_offset = tid * ngeom + geomid;

  const quat qparent = xquat[body_offset];
  const vec3p pparent = xpos[body_offset];

  vec3p rotated_pos = RotVecQuat(geom_pos[geomid], qparent);
  geom_xpos[geom_offset] = pparent + rotated_pos;

  quat qres = MulQuat(qparent, geom_quat[geomid]);
  Quat2Mat(geom_xmat[geom_offset], qres);
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
    mat3p* site_xmat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int siteid = blockIdx.y;
  if (tid >= n) return;

  const int bodyid = __ldg(&site_bodyid[siteid]);
  const unsigned int body_offset = tid * nbody + bodyid;
  const unsigned int site_offset = tid * nsite + siteid;

  const quat qparent = xquat[body_offset];
  const vec3p pparent = xpos[body_offset];

  vec3p rotated_pos = RotVecQuat(site_pos[siteid], qparent);
  site_xpos[site_offset] = pparent + rotated_pos;

  quat qres = MulQuat(qparent, site_quat[siteid]);
  Quat2Mat(site_xmat[site_offset], qres);
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