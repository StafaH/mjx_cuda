#include "smooth.h"
#include "io.h"

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
    float* xpos,
    float* xquat,
    float* xipos,
    float* xmat,
    float* ximat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n) {
    return;
  }

  const unsigned int pos_offset = tid * 3;
  const unsigned int quat_offset = tid * 4;
  const unsigned int mat_offset = tid * 9;
  
  float* batch_xpos = xpos + pos_offset;
  float* batch_xquat = xquat + quat_offset;
  float* batch_xipos = xipos + pos_offset;
  float* batch_xmat = xmat + mat_offset;
  float* batch_ximat = ximat + mat_offset;

  batch_xpos[0] = batch_xpos[1] = batch_xpos[2] = 0.0f;
  batch_xipos[0] = batch_xipos[1] = batch_xipos[2] = 0.0f;
  
  batch_xquat[0] = 1.0f;
  batch_xquat[1] = batch_xquat[2] = batch_xquat[3] = 0.0f;
  
  batch_xmat[0] = batch_xmat[4] = batch_xmat[8] = 1.0f;
  batch_xmat[1] = batch_xmat[2] = batch_xmat[3] = 0.0f;
  batch_xmat[5] = batch_xmat[6] = batch_xmat[7] = 0.0f;
  
  batch_ximat[0] = batch_ximat[4] = batch_ximat[8] = 1.0f;
  batch_ximat[1] = batch_ximat[2] = batch_ximat[3] = 0.0f;
  batch_ximat[5] = batch_ximat[6] = batch_ximat[7] = 0.0f;
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
    const float* body_pos,
    const float* body_quat,
    const float* body_ipos,
    const float* body_iquat,
    const int* jnt_type,
    const int* jnt_qposadr,
    const float* jnt_axis,
    const float* jnt_pos,
    const int* body_tree,
    float* qpos,
    float* mocap_pos,
    float* mocap_quat,
    float* xanchor,
    float* xaxis,
    float* xmat,
    float* xpos,
    float* xquat,
    float* xipos,
    float* ximat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int nodeid = blockIdx.y;

  if (tid >= n) return;

  int bodyid = body_tree[leveladr + nodeid];
  
  const int jntadr = body_jntadr[bodyid];
  const int jntnum = body_jntnum[bodyid];
  
  const unsigned int qpos_offset = tid * nq;
  const unsigned int body_offset = tid * nbody;
  const unsigned int jnt_offset = tid * njnt;
  
  float* batch_xanchor = xanchor + (jnt_offset * 3);
  float* batch_xaxis = xaxis + (jnt_offset * 3);
  float* batch_xmat = xmat + (body_offset * 9);
  float* batch_xpos = xpos + (body_offset * 3);
  float* batch_xquat = xquat + (body_offset * 4);
  float* batch_xipos = xipos + (body_offset * 3);
  float* batch_ximat = ximat + (body_offset * 9);
  
  float lxpos[3] = {0.0f, 0.0f, 0.0f};
  float lxquat[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (jntnum == 0) {
    int pid = body_parentid[bodyid];
    const float* bodypos = body_pos + (3 * bodyid);
    const float* bodyquat = body_quat + (4 * bodyid);
    float* pid_xmat = batch_xmat + (9 * pid);
    float* pid_xpos = batch_xpos + (3 * pid);
    float* pid_xquat = batch_xquat + (4 * pid);

    MulMatVec3(lxpos, pid_xmat, bodypos);
    lxpos[0] += pid_xpos[0];
    lxpos[1] += pid_xpos[1];
    lxpos[2] += pid_xpos[2];
    
    MulQuat(lxquat, pid_xquat, bodyquat);
  } 
  else if (jntnum == 1 && jnt_type[jntadr] == mjJNT_FREE) {
    // Free joint (direct set from qpos)
    int qadr = jnt_qposadr[jntadr];
    float* src_qpos = qpos + qpos_offset + qadr;
    
    lxpos[0] = src_qpos[0];
    lxpos[1] = src_qpos[1];
    lxpos[2] = src_qpos[2];
    lxquat[0] = src_qpos[3];
    lxquat[1] = src_qpos[4];
    lxquat[2] = src_qpos[5];
    lxquat[3] = src_qpos[6];
    
    float norm = sqrtf(lxquat[0]*lxquat[0] + lxquat[1]*lxquat[1] + 
                      lxquat[2]*lxquat[2] + lxquat[3]*lxquat[3]);
    float invNorm = 1.0f / norm;
    lxquat[0] *= invNorm;
    lxquat[1] *= invNorm;
    lxquat[2] *= invNorm;
    lxquat[3] *= invNorm;

    float* jnt_anchor = batch_xanchor + (3 * jntadr);
    float* jnt_ax = batch_xaxis + (3 * jntadr);
    const float* joint_axis = jnt_axis + (3 * jntadr);
    
    jnt_anchor[0] = lxpos[0];
    jnt_anchor[1] = lxpos[1];
    jnt_anchor[2] = lxpos[2];
    jnt_ax[0] = joint_axis[0];
    jnt_ax[1] = joint_axis[1];
    jnt_ax[2] = joint_axis[2];
  } 
  else {
    int pid = body_parentid[bodyid];
    const float* bodypos = body_pos + (3 * bodyid);
    const float* bodyquat = body_quat + (4 * bodyid);
    float* pid_xmat = batch_xmat + (9 * pid);
    float* pid_xpos = batch_xpos + (3 * pid);
    float* pid_xquat = batch_xquat + (4 * pid);

    MulMatVec3(lxpos, pid_xmat, bodypos);
    lxpos[0] += pid_xpos[0];
    lxpos[1] += pid_xpos[1];
    lxpos[2] += pid_xpos[2];
    
    MulQuat(lxquat, pid_xquat, bodyquat);

    float* curr_qpos = qpos + qpos_offset;
    
    for (int j = 0; j < jntnum; j++) {
      int jid = jntadr + j;
      int qadr = jnt_qposadr[jid];
      int jtype = jnt_type[jid];
      
      float vec[3];
      const float* jnt_pos_ptr = jnt_pos + (3 * jid);
      float* jnt_anchor = batch_xanchor + (3 * jid);
      float* jnt_ax = batch_xaxis + (3 * jid);
      const float* joint_axis = jnt_axis + (3 * jid);
      
      RotVecQuat(vec, jnt_pos_ptr, lxquat);
      
      jnt_anchor[0] = vec[0] + lxpos[0];
      jnt_anchor[1] = vec[1] + lxpos[1];
      jnt_anchor[2] = vec[2] + lxpos[2];

      RotVecQuat(jnt_ax, joint_axis, lxquat);

      switch (jtype) {
        case mjJNT_SLIDE: {
          float qpos_diff = curr_qpos[qadr] - qpos0[qadr];
          lxpos[0] += jnt_ax[0] * qpos_diff;
          lxpos[1] += jnt_ax[1] * qpos_diff;
          lxpos[2] += jnt_ax[2] * qpos_diff;
          break;
        }

        case mjJNT_BALL: {
          float qloc[4];
          qloc[0] = curr_qpos[qadr];
          qloc[1] = curr_qpos[qadr+1];
          qloc[2] = curr_qpos[qadr+2];
          qloc[3] = curr_qpos[qadr+3];
          
          float new_quat[4];
          MulQuat(new_quat, lxquat, qloc);
          
          lxquat[0] = new_quat[0];
          lxquat[1] = new_quat[1];
          lxquat[2] = new_quat[2];
          lxquat[3] = new_quat[3];
          
          RotVecQuat(vec, jnt_pos_ptr, lxquat);
          lxpos[0] = jnt_anchor[0] - vec[0];
          lxpos[1] = jnt_anchor[1] - vec[1];
          lxpos[2] = jnt_anchor[2] - vec[2];
          break;
        }

        case mjJNT_HINGE: {
          float angle = curr_qpos[qadr] - qpos0[qadr];
          float qloc[4];
          
          float s = sinf(angle * 0.5f);
          qloc[0] = cosf(angle * 0.5f);
          qloc[1] = joint_axis[0] * s;
          qloc[2] = joint_axis[1] * s;
          qloc[3] = joint_axis[2] * s;
          
          float new_quat[4];
          MulQuat(new_quat, lxquat, qloc);
          
          lxquat[0] = new_quat[0];
          lxquat[1] = new_quat[1];
          lxquat[2] = new_quat[2];
          lxquat[3] = new_quat[3];
          
          RotVecQuat(vec, jnt_pos_ptr, lxquat);
          lxpos[0] = jnt_anchor[0] - vec[0];
          lxpos[1] = jnt_anchor[1] - vec[1];
          lxpos[2] = jnt_anchor[2] - vec[2];
          break;
        }
      }
    }
  }

  float norm = sqrtf(lxquat[0]*lxquat[0] + lxquat[1]*lxquat[1] + 
                     lxquat[2]*lxquat[2] + lxquat[3]*lxquat[3]);
  float invNorm = 1.0f / norm;
  lxquat[0] *= invNorm;
  lxquat[1] *= invNorm;
  lxquat[2] *= invNorm;
  lxquat[3] *= invNorm;

  float* body_xquat = batch_xquat + (4 * bodyid);
  float* body_xpos = batch_xpos + (3 * bodyid);
  float* body_xmat = batch_xmat + (9 * bodyid);
  
  body_xquat[0] = lxquat[0];
  body_xquat[1] = lxquat[1];
  body_xquat[2] = lxquat[2];
  body_xquat[3] = lxquat[3];
  
  body_xpos[0] = lxpos[0];
  body_xpos[1] = lxpos[1];
  body_xpos[2] = lxpos[2];
  
  Quat2Mat(body_xmat, lxquat);

  float vec[3];
  const float* body_ipos_ptr = body_ipos + (3 * bodyid);
  float* body_xipos = batch_xipos + (3 * bodyid);
  
  RotVecQuat(vec, body_ipos_ptr, lxquat);
  body_xipos[0] = vec[0] + lxpos[0];
  body_xipos[1] = vec[1] + lxpos[1];
  body_xipos[2] = vec[2] + lxpos[2];

  float quat[4];
  const float* body_iquat_ptr = body_iquat + (4 * bodyid);
  float* body_ximat = batch_ximat + (9 * bodyid);
  
  MulQuat(quat, lxquat, body_iquat_ptr);
  Quat2Mat(body_ximat, quat);
}

__global__ void GeomLocalToGlobalKernel(
    unsigned int n,
    unsigned int nbody,
    unsigned int ngeom,
    const int* geom_bodyid,
    const float* geom_pos,
    const float* geom_quat,
    const float* xpos,
    const float* xquat,
    float* geom_xpos,
    float* geom_xmat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n) return;

  const unsigned int body_offset = tid * nbody;
  const unsigned int geom_offset = tid * ngeom;
  
  const float* batch_xpos = xpos + body_offset * 3;
  const float* batch_xquat = xquat + body_offset * 4;
  float* batch_geom_xpos = geom_xpos + geom_offset * 3;
  float* batch_geom_xmat = geom_xmat + geom_offset * 9;

  for (int i = 0; i < ngeom; i++) {
    int bodyid = geom_bodyid[i];
    const float* body_xquat = batch_xquat + 4 * bodyid;
    const float* body_xpos = batch_xpos + 3 * bodyid;
    
    float vec[3];
    RotVecQuat(vec, geom_pos + 3 * i, body_xquat);
    
    float* geom_xpos_i = batch_geom_xpos + 3 * i;
    geom_xpos_i[0] = vec[0] + body_xpos[0];
    geom_xpos_i[1] = vec[1] + body_xpos[1];
    geom_xpos_i[2] = vec[2] + body_xpos[2];

    float quat[4];
    MulQuat(quat, body_xquat, geom_quat + 4 * i);
    Quat2Mat(batch_geom_xmat + 9 * i, quat);
  }
}

__global__ void SiteLocalToGlobalKernel(
    unsigned int n,
    unsigned int nbody,
    unsigned int nsite,
    const int* site_bodyid,
    const float* site_pos,
    const float* site_quat,
    const float* xpos,
    const float* xquat,
    float* site_xpos,
    float* site_xmat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n) return;

  const unsigned int body_offset = tid * nbody;
  const unsigned int site_offset = tid * nsite;
  
  const float* batch_xpos = xpos + body_offset * 3;
  const float* batch_xquat = xquat + body_offset * 4;
  float* batch_site_xpos = site_xpos + site_offset * 3;
  float* batch_site_xmat = site_xmat + site_offset * 9;

  for (int i = 0; i < nsite; i++) {
    int bodyid = site_bodyid[i];
    const float* body_xquat = batch_xquat + 4 * bodyid;
    const float* body_xpos = batch_xpos + 3 * bodyid;
    
    float vec[3];
    RotVecQuat(vec, site_pos + 3 * i, body_xquat);
    
    float* site_xpos_i = batch_site_xpos + 3 * i;
    site_xpos_i[0] = vec[0] + body_xpos[0];
    site_xpos_i[1] = vec[1] + body_xpos[1];
    site_xpos_i[2] = vec[2] + body_xpos[2];

    float quat[4];
    MulQuat(quat, body_xquat, site_quat + 4 * i);
    Quat2Mat(batch_site_xmat + 9 * i, quat);
  }
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