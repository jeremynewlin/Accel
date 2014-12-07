#ifndef KD_TREE_GPU_H
#define KD_TREE_GPU_H

#include "KDTreeStructs.h"
#include "KDTreeCPU.h"


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

	void buildTree( KDTreeNode *curr_node );
};

#endif