#pragma once

#include <cuda_runtime.h>

struct CudaModel {
  int nq;
  int njnt;
  int nbody;
  int ngeom;
  int nsite;
  int nmocap;

  float* qpos0;
  int* body_tree;
  int* body_leveladr;
  int* body_jntadr;
  int* body_jntnum;
  int* body_parentid;
  int* body_mocapid;
  float* body_pos;
  float* body_quat;
  float* body_ipos;
  float* body_iquat;
  int* jnt_type;
  int* jnt_qposadr;
  float* jnt_axis;
  float* jnt_pos;
  int* geom_bodyid;
  float* geom_pos;
  float* geom_quat;
  int* site_bodyid;
  float* site_pos;
  float* site_quat;

  CudaModel() : nq(0), njnt(0), nbody(0), ngeom(0), nsite(0), nmocap(0),
                qpos0(nullptr), body_tree(nullptr), body_leveladr(nullptr), body_jntadr(nullptr), body_jntnum(nullptr),
                body_parentid(nullptr), body_mocapid(nullptr), body_pos(nullptr),
                body_quat(nullptr), body_ipos(nullptr), body_iquat(nullptr),
                jnt_type(nullptr), jnt_qposadr(nullptr), jnt_axis(nullptr),
                jnt_pos(nullptr), geom_bodyid(nullptr), geom_pos(nullptr),
                geom_quat(nullptr), site_bodyid(nullptr), site_pos(nullptr),
                site_quat(nullptr) {}

  ~CudaModel() {
    if (qpos0) cudaFree(qpos0);
    if (body_jntadr) cudaFree(body_jntadr);
    if (body_jntnum) cudaFree(body_jntnum);
    if (body_parentid) cudaFree(body_parentid);
    if (body_mocapid) cudaFree(body_mocapid);
    if (body_pos) cudaFree(body_pos);
    if (body_quat) cudaFree(body_quat);
    if (body_ipos) cudaFree(body_ipos);
    if (body_iquat) cudaFree(body_iquat);
    if (jnt_type) cudaFree(jnt_type);
    if (jnt_qposadr) cudaFree(jnt_qposadr);
    if (jnt_axis) cudaFree(jnt_axis);
    if (jnt_pos) cudaFree(jnt_pos);
    if (geom_bodyid) cudaFree(geom_bodyid);
    if (geom_pos) cudaFree(geom_pos);
    if (geom_quat) cudaFree(geom_quat);
    if (site_bodyid) cudaFree(site_bodyid);
    if (site_pos) cudaFree(site_pos);
    if (site_quat) cudaFree(site_quat);
  }
};

struct CudaData {
  int nq;
  int nmocap;
  int nbody;
  int ngeom;
  int nsite;
  int batch_size;

  float* qpos;
  float* mocap_pos;
  float* mocap_quat;
  float* xanchor;
  float* xaxis;
  float* xmat;
  float* xpos;
  float* xquat;
  float* xipos;
  float* ximat;
  float* geom_xpos;
  float* geom_xmat;
  float* site_xpos;
  float* site_xmat;

  CudaData() : nq(0), nmocap(0), nbody(0), ngeom(0), nsite(0), batch_size(0),
               qpos(nullptr), mocap_pos(nullptr), mocap_quat(nullptr),
               xanchor(nullptr), xaxis(nullptr), xmat(nullptr),
               xpos(nullptr), xquat(nullptr), xipos(nullptr),
               ximat(nullptr), geom_xpos(nullptr), geom_xmat(nullptr),
               site_xpos(nullptr), site_xmat(nullptr) {}

  ~CudaData() {
    if (qpos) cudaFree(qpos);
    if (mocap_pos) cudaFree(mocap_pos);
    if (mocap_quat) cudaFree(mocap_quat);
    if (xanchor) cudaFree(xanchor);
    if (xaxis) cudaFree(xaxis);
    if (xmat) cudaFree(xmat);
    if (xpos) cudaFree(xpos);
    if (xquat) cudaFree(xquat);
    if (xipos) cudaFree(xipos);
    if (ximat) cudaFree(ximat);
    if (geom_xpos) cudaFree(geom_xpos);
    if (geom_xmat) cudaFree(geom_xmat); 
    if (site_xpos) cudaFree(site_xpos); 
    if (site_xmat) cudaFree(site_xmat);
  }
};