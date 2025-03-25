#include "smooth.h"
#include "io.h"

void LaunchKinematicsKernel(
    unsigned int batch_size,
    CudaModel* cm,
    CudaData* cd) {

  int threadsPerBlock = 256;
  int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

  // Launch root kernel
  RootKernel<<<numBlocks, threadsPerBlock>>>(
      batch_size,
      cd->xpos,
      cd->xquat,
      cd->xipos,
      cd->xmat,
      cd->ximat);

  // Launch level kernels
  // for (int i = 1; i < cm->nbody; i++) {
  //   int beg = cm->body_leveladr[i];
  //   int end = (i == cm->nbody - 1) ? cm->nbody : cm->body_leveladr[i + 1];
  //   int numBodiesInLevel = end - beg;
    
  //   dim3 gridDim((batch_size + threadsPerBlock - 1) / threadsPerBlock, numBodiesInLevel);
  //   LevelKernel<<<gridDim, threadsPerBlock>>>(
  //       batch_size,
  //       beg,
  //       cm->nq,
  //       cm->njnt,
  //       cm->nbody,
  //       cm->qpos0,
  //       cm->body_jntadr,
  //       cm->body_jntnum,
  //       cm->body_parentid,
  //       cm->body_mocapid,
  //       cm->body_pos,
  //       cm->body_quat,
  //       cm->body_ipos,
  //       cm->body_iquat,
  //       cm->jnt_type,
  //       cm->jnt_qposadr,
  //       cm->jnt_axis,
  //       cm->jnt_pos,
  //       cm->body_tree,
  //       cd->qpos,
  //       cd->mocap_pos,
  //       cd->mocap_quat,
  //       cd->xanchor,
  //       cd->xaxis,
  //       cd->xmat,
  //       cd->xpos,
  //       cd->xquat,
  //       cd->xipos,
  //       cd->ximat);
  // }

  // // Launch geometry kernel if needed
  // if (cm->ngeom > 0) {
  //   dim3 gridDim((batch_size + threadsPerBlock - 1) / threadsPerBlock, cm->ngeom);
  //   GeomLocalToGlobalKernel<<<gridDim, threadsPerBlock>>>(
  //       batch_size,
  //       cm->nbody,
  //       cm->ngeom,
  //       cm->geom_bodyid,
  //       cm->geom_pos,
  //       cm->geom_quat,
  //       cd->xpos,
  //       cd->xquat,
  //       cd->geom_xpos,
  //       cd->geom_xmat);
  // }

  // // Launch site kernel if needed
  // if (cm->nsite > 0) {
  //   dim3 gridDim((batch_size + threadsPerBlock - 1) / threadsPerBlock, cm->nsite);
  //   SiteLocalToGlobalKernel<<<gridDim, threadsPerBlock>>>(
  //       batch_size,
  //       cm->nbody,
  //       cm->nsite,
  //       cm->site_bodyid,
  //       cm->site_pos,
  //       cm->site_quat,
  //       cd->xpos,
  //       cd->xquat,
  //       cd->site_xpos,
  //       cd->site_xmat);
  // }
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

  // batch index into data
  xpos = xpos + tid * 3;
  xquat = xquat + tid * 4;
  xipos = xipos + tid * 3;
  xmat = xmat + tid * 9;
  ximat = ximat + tid * 9;

  // set world position and orientation
  Zero(xpos, 3);
  Zero(xquat, 4);
  Zero(xipos, 3);
  Zero(xmat, 9);
  Zero(ximat, 9);

  xquat[0] = 1.0f;
  xmat[0] = xmat[4] = xmat[8] = 1.0f;
  ximat[0] = ximat[4] = ximat[8] = 1.0f;
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
    const float* qpos,
    const float* mocap_pos,
    const float* mocap_quat,
    float* xanchor,
    float* xaxis,
    float* xmat,
    float* xpos,
    float* xquat,
    float* xipos,
    float* ximat) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int nodeid = blockIdx.y;

  if (tid >= n) {
    return;
  }

  // Get bodyid from body_tree
  int bodyid = body_tree[leveladr + nodeid];
  int jntadr = body_jntadr[bodyid];
  int jntnum = body_jntnum[bodyid];
  float lxpos[3], lxquat[4];

  // batch index into data
  qpos = qpos + tid * nq;
  mocap_pos = mocap_pos + tid * 3;
  mocap_quat = mocap_quat + tid * 4;

  xanchor = xanchor + tid * njnt * 3;
  xaxis = xaxis + tid * njnt * 3;
  xmat = xmat + tid * nbody * 9;
  xpos = xpos + tid * nbody * 3;
  xquat = xquat + tid * nbody * 4;
  xipos = xipos + tid * nbody * 3;
  ximat = ximat + tid * nbody * 9;

  if (jntnum == 0) {
    // no joints - apply fixed translation and rotation relative to parent
    int pid = body_parentid[bodyid];
    const float* bodypos = body_pos + 3 * bodyid;
    const float* bodyquat = body_quat + 4 * bodyid;

    MulMatVec3(lxpos, xmat + 9 * pid, bodypos);
    AddTo(lxpos, xpos + 3 * pid, 3);
    MulQuat(lxquat, xquat + 4 * pid, bodyquat);
  } else if (jntnum == 1 && jnt_type[jntadr] == mjJNT_FREE) {
    // free joint
    int qadr = jnt_qposadr[jntadr];
    Copy(lxpos, qpos + qadr, 3);
    Copy(lxquat, qpos + qadr + 3, 4);
    Normalize(lxquat, 4);

    // assign xanchor and xaxis
    Copy(xanchor + 3 * jntadr, lxpos, 3);
    Copy(xaxis + 3 * jntadr, jnt_axis + 3 * jntadr, 3);
  } else {
    // regular or no joints
    int pid = body_parentid[bodyid];
    const float* bodypos = body_pos + 3 * bodyid;
    const float* bodyquat = body_quat + 4 * bodyid;

    // apply fixed translation and rotation relative to parent
    MulMatVec3(lxpos, xmat + 9 * pid, bodypos);
    AddTo(lxpos, xpos + 3 * pid, 3);
    MulQuat(lxquat, xquat + 4 * pid, bodyquat);

    // accumulate joints
    for (int j = 0; j < jntnum; j++) {
      int jid = jntadr + j;
      int qadr = jnt_qposadr[jid];
      int jtype = jnt_type[jid];

      // compute axis in global frame
      RotVecQuat(xaxis + 3 * jid, jnt_axis + 3 * jid, lxquat);

      // compute anchor in global frame
      RotVecQuat(xanchor + 3 * jid, jnt_pos + 3 * jid, lxquat);
      AddTo(xanchor + 3 * jid, lxpos, 3);

      // apply joint transformation
      switch (jtype) {
        case mjJNT_SLIDE:
          AddToScl(lxpos, xaxis + 3 * jid, qpos[qadr] - qpos0[qadr], 3);
          break;

        case mjJNT_BALL: {
          float qloc[4];
          Copy(qloc, qpos + qadr, 4);
          Normalize(qloc, 4);
          MulQuat(lxquat, lxquat, qloc);
          float vec[3];
          RotVecQuat(vec, jnt_pos + 3 * jid, lxquat);
          Sub(lxpos, xanchor + 3 * jid, vec, 3);
          break;
        }

        case mjJNT_HINGE: {
          float qloc[4];
          AxisAngle2Quat(qloc, jnt_axis + 3 * jid, qpos[qadr] - qpos0[qadr]);
          MulQuat(lxquat, lxquat, qloc);
          float vec[3];
          RotVecQuat(vec, jnt_pos + 3 * jid, lxquat);
          Sub(lxpos, xanchor + 3 * jid, vec, 3);
          break;
        }

        default:
          break;
      }
    }
  }

  // assign xquat and xpos, construct xmat
  Normalize(lxquat, 4);
  Copy(xquat + 4 * bodyid, lxquat, 4);
  Copy(xpos + 3 * bodyid, lxpos, 3);
  Quat2Mat(xmat + 9 * bodyid, lxquat);

  // compute xipos and ximat
  float vec[3];
  RotVecQuat(vec, body_ipos + 3 * bodyid, lxquat);
  AddTo(xipos + 3 * bodyid, vec, 3);

  float quat[4];
  MulQuat(quat, lxquat, body_iquat + 4 * bodyid);
  Quat2Mat(ximat + 9 * bodyid, quat);
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

  if (tid >= n) {
    return;
  }

  // batch index into data
  xpos = xpos + tid * nbody * 3;
  xquat = xquat + tid * nbody * 4;
  geom_xpos = geom_xpos + tid * ngeom * 3;
  geom_xmat = geom_xmat + tid * ngeom * 9;

  for (int i = 0; i < ngeom; i++) {
    int bodyid = geom_bodyid[i];
    float vec[3];
    RotVecQuat(vec, geom_pos + 3 * i, xquat + 4 * bodyid);
    AddTo(geom_xpos + 3 * i, vec, 3);

    float quat[4];
    MulQuat(quat, xquat + 4 * bodyid, geom_quat + 4 * i);
    Quat2Mat(geom_xmat + 9 * i, quat);
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

  if (tid >= n) {
    return;
  }

  // batch index into data
  xpos = xpos + tid * nbody * 3;
  xquat = xquat + tid * nbody * 4;
  site_xpos = site_xpos + tid * nsite * 3;
  site_xmat = site_xmat + tid * nsite * 9;

  for (int i = 0; i < nsite; i++) {
    int bodyid = site_bodyid[i];
    float vec[3];
    RotVecQuat(vec, site_pos + 3 * i, xquat + 4 * bodyid);
    AddTo(site_xpos + 3 * i, vec, 3);

    float quat[4];
    MulQuat(quat, xquat + 4 * bodyid, site_quat + 4 * i);
    Quat2Mat(site_xmat + 9 * i, quat);
  }
}