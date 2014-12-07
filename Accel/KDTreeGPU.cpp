#include "KDTreeGPU.h"
#include <iostream>


////////////////////////////////////////////////////
// Construtor/destructor.
////////////////////////////////////////////////////

KDTreeGPU::KDTreeGPU( KDTreeCPU *kd_tree_cpu )
{
	num_nodes = kd_tree_cpu->getNumNodes();
	cpu_tree_root = kd_tree_cpu->getRootNode();
	root_index = cpu_tree_root->id;

	// Allocate memory for all nodes in GPU kd-tree.
	tree_nodes = new KDTreeNodeGPU[num_nodes];

	// Populate tree_nodes and tri_index_list.
	tri_index_list.clear();
	buildTree( cpu_tree_root );
}

KDTreeGPU::~KDTreeGPU()
{
	delete[] tree_nodes;
}


////////////////////////////////////////////////////
// Recursive method to build up GPU kd-tree structure from CPU kd-tree structure.
// This method populates tree_nodes, an array of KDTreeNodeGPUs and
// tri_index_list, a list of triangle indices for all leaf nodes to be sent to the device.
////////////////////////////////////////////////////
void KDTreeGPU::buildTree( KDTreeNode *curr_node )
{
	// Get index of node in CPU kd-tree.
    int index = curr_node->id;
	
	// Start building GPU kd-tree node from current CPU kd-tree node.
	tree_nodes[index].bbox = curr_node->bbox;
	tree_nodes[index].split_plane_axis = curr_node->split_plane_axis;
	tree_nodes[index].split_plane_value = curr_node->split_plane_value;

	// Leaf node.
    if ( curr_node->is_leaf_node ) {
        tree_nodes[index].num_tris = curr_node->num_tris;
        tree_nodes[index].first_tri_index = tri_index_list.size(); // tri_index_list initially contains 0 elements.

		// Add triangles to tri_index_list as each leaf node is processed.
        for ( int i = 0; i < curr_node->num_tris; ++i ) {
            tri_index_list.push_back( curr_node->tri_indices[i] );
        }

		// Set neighboring node indices for GPU node using ropes in CPU node.
        for ( int i = 0; i < 6; ++i ) {
            if ( curr_node->ropes[i] ) {
                tree_nodes[index].neighbor_node_indices[i] = curr_node->ropes[i]->id;
			}
        }
    }
    else {
        if ( curr_node->left ) {
			// Set child node index for current node and recurse.
			tree_nodes[index].left_child_index = curr_node->left->id;
            buildTree( curr_node->left );
        }
		if ( curr_node->right ) {
			// Set child node index for current node and recurse.
			tree_nodes[index].right_child_index = curr_node->right->id;
            buildTree( curr_node->right );
        }
    }
}


////////////////////////////////////////////////////
// Debug methods.
////////////////////////////////////////////////////

void KDTreeGPU::printGPUNodeDataWithCorrespondingCPUNodeData( KDTreeNode *curr_node, bool pause_on_each_node )
{
	curr_node->prettyPrint();
	tree_nodes[curr_node->id].prettyPrint();

	if ( pause_on_each_node ) {
		std::cin.ignore();
	}

    if ( curr_node->left ) {
        printGPUNodeDataWithCorrespondingCPUNodeData( curr_node->left, pause_on_each_node );
    }
	if ( curr_node->right ) {
        printGPUNodeDataWithCorrespondingCPUNodeData( curr_node->right, pause_on_each_node );
    }
}