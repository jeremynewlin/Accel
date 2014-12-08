#ifndef KD_TRAVERSAL_ON_GPU_H
#define KD_TRAVERSAL_ON_GPU_H

#include <cuda.h>
#include "../glm/glm.hpp"
#include "Camera.h"
#include "mesh.h"
#include "KDTreeGPU.h"
#include "boundingBox.h"


// Kernel wrapper function.
glm::vec3* cudaRayCastObj( Camera *camera, mesh *obj_mesh, KDTreeGPU *kd_tree, bool use_brute_force_approach=false );

// CUDA kernels.
__global__
void performGPURaycast( glm::vec3 *image_buffer,
						glm::vec2 cam_reso, glm::vec3 cam_pos, glm::vec3 cam_m, glm::vec3 cam_h, glm::vec3 cam_v,
						glm::vec3 *mesh_tris, glm::vec3 *mesh_verts,
						int kd_tree_root_index, KDTreeNodeGPU *kd_tree_nodes, int *kd_tree_tri_indices );

__global__
void performBruteForceGPURaycast( glm::vec3 *image_buffer,
								  glm::vec2 cam_reso, glm::vec3 cam_pos, glm::vec3 cam_m, glm::vec3 cam_h, glm::vec3 cam_v,
								  boundingBox bbox, int num_tris, glm::vec3 *mesh_tris, glm::vec3 *mesh_verts );

// kd-tree traversal method on the GPU.
__device__ bool gpuStacklessGPUIntersect( const glm::vec3 &ray_o, const glm::vec3 &ray_dir,
										  int root_index, KDTreeNodeGPU *tree_nodes, int *kd_tri_index_list,
										  glm::vec3 *tris, glm::vec3 *verts,
										  float &t, glm::vec3 &hit_point, glm::vec3 &normal );

// KDTreeNodeGPU methods.
__device__ bool gpuIsPointToLeftOfSplittingPlane( KDTreeNodeGPU node, const glm::vec3 &p );
__device__ int gpuGetNeighboringNodeIndex( KDTreeNodeGPU node, glm::vec3 p );

// Intersection methods.
__device__ bool gpuAABBIntersect( boundingBox bbox, glm::vec3 ray_o, glm::vec3 ray_dir, float &t_near, float &t_far );
__device__ bool gpuTriIntersect( glm::vec3 ray_o, glm::vec3 ray_dir, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t, glm::vec3 &normal );
__device__ glm::vec3 gpuComputeTriNormal( const glm::vec3&, const glm::vec3&, const glm::vec3& );

__device__ float findMax( const float &a, const float &b );
__device__ float findMin( const float &a, const float &b );


#endif