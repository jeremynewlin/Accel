#ifndef KD_TREE_GPU_H
#define KD_TREE_GPU_H

#include "KDTreeStructs.h"
#include "KDTreeCPU.h"
#include <cuda_runtime.h>


class KDTreeGPU
{
public:
	KDTreeGPU( KDTreeCPU *kd_tree_cpu );
	~KDTreeGPU( void );

	// Debug method.
	void printGPUNodeDataWithCorrespondingCPUNodeData( KDTreeNode *curr_node, bool pause_on_each_node=false );

private:
	KDTreeNodeGPU *tree_nodes;
	std::vector<int> tri_index_list;

	int num_nodes;
	int root_index;

	KDTreeNode *cpu_tree_root;

	// Input mesh variables.
	int num_verts, num_tris;
	glm::vec3 *verts, *tris;

	void buildTree( KDTreeNode *curr_node );
};


// kd-tree traversal method on the GPU.
__device__ bool intersect( const glm::vec3 &ray_o, const glm::vec3 &ray_dir,
						   int root_index, KDTreeNodeGPU *tree_nodes, int *kd_tri_index_list,
						   glm::vec3 *tris, glm::vec3 *verts,
						   float &t, glm::vec3 &hit_point, glm::vec3 &normal );

#endif