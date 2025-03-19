#include "smooth.h"
#include "io.h"
void LaunchKinematicsKernel(
    unsigned int batch_size,
    CudaModel* cm,
    CudaData* cd) {

  int threadsPerBlock = 256;
  int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

  KinematicsKernel<<<numBlocks, threadsPerBlock>>>(
      batch_size,
      cm->nq,
      cm->njnt,
      cm->nbody,
      cm->ngeom,
      cm->nsite,
      cm->nmocap,
      cm->qpos0,
      cm->body_jntadr,
      cm->body_jntnum,
      cm->body_parentid,
      cm->body_mocapid,
      cm->body_pos,
      cm->body_quat,
      cm->jnt_type,
      cm->jnt_qposadr,
      cm->jnt_axis,
      cm->jnt_pos,
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


__global__ void KinematicsKernel(
    unsigned int n,
    unsigned int nq,
    unsigned int njnt,
    unsigned int nbody,
    unsigned int ngeom,
    unsigned int nsite,
    unsigned int nmocap,
    const float* qpos0,
    const int* body_jntadr,
    const int* body_jntnum,
    const int* body_parentid,
    const int* body_mocapid,
    const float* body_pos,
    const float* body_quat,
    const int* jnt_type,
    const int* jnt_qposadr,
    const float* jnt_axis,
    const float* jnt_pos,
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

  if (tid >= n) {
    return;
  }

  // batch index into mjx.Data
  qpos = qpos + tid * nq;
  mocap_pos = mocap_pos + tid * nmocap * 3;
  mocap_quat = mocap_quat + tid * nmocap * 4;

  xanchor = xanchor + tid * njnt * 3;
  xaxis = xaxis + tid * njnt * 3;
  xmat = xmat + tid * nbody * 9;
  xpos = xpos + tid * nbody * 3;
  xquat = xquat + tid * nbody * 4;
  xipos = xipos + tid * nbody * 3;
  ximat = ximat + tid * nbody * 9;

  // set world position and orientation
  Zero(xpos, 3);
  Zero(xquat, 4);
  Zero(xipos, nbody * 3);
  Zero(xmat, nbody * 9);
  Zero(ximat, nbody * 9);

  xquat[0] = 1.0f;
  xmat[0] = xmat[4] = xmat[8] = 1.0f;
  ximat[0] = ximat[4] = ximat[8] = 1.0f;

  // TODO(erikfrey): bodies in the same tree depth can be processed in parallel
  // tried threading over bodies and it was actually significantly slower -
  // need to investigate further

  // compute global cartesian positions and orientations of all bodies
  for (int i = 1; i < nbody; i++) {
    float lxpos[3], lxquat[4];
    int jntadr = body_jntadr[i];
    int jntnum = body_jntnum[i];

    // free joint
    if (jntnum == 1 && jnt_type[jntadr] == mjJNT_FREE) {
      // get qpos address
      int qadr = jnt_qposadr[jntadr];

      // copy pos and quat from qpos
      Copy(lxpos, qpos + qadr, 3);
      Copy(lxquat, qpos + qadr + 3, 4);
      Normalize(lxquat, 4);

      // assign xanchor and xaxis
      Copy(xanchor + 3 * jntadr, lxpos, 3);
      Copy(xaxis + 3 * jntadr, jnt_axis + 3 * jntadr, 3);
    } else {  // regular or no joint
      int pid = body_parentid[i];

      // get body pos and quat: from model or mocap
      const float *bodypos, *bodyquat;
      float quat[4];
      if (body_mocapid[i] >= 0) {
        bodypos = mocap_pos + 3 * body_mocapid[i];
        Copy(quat, mocap_quat + 4 * body_mocapid[i], 4);
        Normalize(quat, 4);
        bodyquat = quat;
      } else {
        bodypos = body_pos + 3 * i;
        bodyquat = body_quat + 4 * i;
      }

      // apply fixed translation and rotation relative to parent
      if (pid) {
        MulMatVec3(lxpos, xmat + 9 * pid, bodypos);
        AddTo(lxpos, xpos + 3 * pid, 3);
        MulQuat(lxquat, xquat + 4 * pid, bodyquat);
      } else {
        // parent is the world
        Copy(lxpos, bodypos, 3);
        Copy(lxquat, bodyquat, 4);
      }

      // accumulate joints, compute xpos and xquat for this body
      float lxanchor[3], lxaxis[3];
      for (int j = 0; j < jntnum; j++) {
        // get joint id, qpos address, joint type
        int jid = jntadr + j;
        int qadr = jnt_qposadr[jid];
        int jtype = jnt_type[jid];

        // compute axis in global frame; ball jnt_axis is (0,0,1), set by
        // compiler
        RotVecQuat(lxaxis, jnt_axis + 3 * jid, lxquat);

        // compute anchor in global frame
        RotVecQuat(lxanchor, jnt_pos + 3 * jid, lxquat);
        AddTo(lxanchor, lxpos, 3);

        // apply joint transformation
        switch (jtype) {
          case mjJNT_SLIDE:
            AddToScl(lxpos, lxaxis, qpos[qadr] - qpos0[qadr], 3);
            break;

          case mjJNT_BALL:
          case mjJNT_HINGE: {
            // compute local quaternion rotation
            float qloc[4];
            if (jtype == mjJNT_BALL) {
              Copy(qloc, qpos + qadr, 4);
              Normalize(qloc, 4);
            } else {
              AxisAngle2Quat(qloc, jnt_axis + 3 * jid,
                             qpos[qadr] - qpos0[qadr]);
            }

            // apply rotation
            MulQuat(lxquat, lxquat, qloc);

            // correct for off-center rotation
            float vec[3];
            RotVecQuat(vec, jnt_pos + 3 * jid, lxquat);
            Sub(lxpos, lxanchor, vec, 3);
          } break;

          default:
            // TODO: whatever cuda error semantics are
            // mjERROR("unknown joint type %d", jtype);  // SHOULD NOT OCCUR
            break;
        }

        // assign xanchor and xaxis
        Copy(xanchor + 3 * jid, lxanchor, 3);
        Copy(xaxis + 3 * jid, lxaxis, 3);
      }
    }

    // assign xquat and xpos, construct xmat
    Normalize(lxquat, 4);
    Copy(xquat + 4 * i, lxquat, 4);
    Copy(xpos + 3 * i, lxpos, 3);
    Quat2Mat(xmat + 9 * i, lxquat);
  }
}