#include "KDTreeGPU.h"
#include <iostream>
#include "Intersections.h"


////////////////////////////////////////////////////
// Construtor/destructor.
////////////////////////////////////////////////////

KDTreeGPU::KDTreeGPU( KDTreeCPU *kd_tree_cpu )
{
	num_nodes = kd_tree_cpu->getNumNodes();
	cpu_tree_root = kd_tree_cpu->getRootNode();
	root_index = cpu_tree_root->id;

	num_verts = kd_tree_cpu->getMeshNumVerts();
	num_tris = kd_tree_cpu->getMeshNumTris();

	glm::vec3 *tmp_verts = kd_tree_cpu->getMeshVerts();
	verts = new glm::vec3[num_verts];
	for ( int i = 0; i < num_verts; ++i ) {
		verts[i] = tmp_verts[i];
	}

	glm::vec3 *tmp_tris = kd_tree_cpu->getMeshTris();
	tris = new glm::vec3[num_tris];
	for ( int i = 0; i < num_tris; ++i ) {
		tris[i] = tmp_tris[i];
	}


	// Allocate memory for all nodes in GPU kd-tree.
	tree_nodes = new KDTreeNodeGPU[num_nodes];

	// Populate tree_nodes and tri_index_list.
	tri_index_list.clear();
	buildTree( cpu_tree_root );
}

KDTreeGPU::~KDTreeGPU()
{
	if ( num_verts > 0 ) {
		delete[] verts;
	}

	if ( num_tris > 0 ) {
		delete[] tris;
	}

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
	tree_nodes[index].is_leaf_node = curr_node->is_leaf_node;

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


////////////////////////////////////////////////////
// GPU stackless kd-tree traversal method to be called from CUDA kernel.
////////////////////////////////////////////////////
__device__
bool intersect( const glm::vec3 &ray_o, const glm::vec3 &ray_dir,
				int root_index, KDTreeNodeGPU *tree_nodes, int *kd_tri_index_list,
				glm::vec3 *tris, glm::vec3 *verts,
				float &t, glm::vec3 &hit_point, glm::vec3 &normal )
{
    KDTreeNodeGPU curr_node = tree_nodes[root_index];

	// Perform ray/AABB intersection test.
	float t_entry, t_exit;
	bool intersects_root_node_bounding_box = Intersections::aabbIntersect( curr_node.bbox, ray_o, ray_dir, t_entry, t_exit );

	if ( !intersects_root_node_bounding_box ) {
		return false;
	}

	bool intersection_detected = false;

	float t_entry_prev = -INFINITY;
	while ( t_entry < t_exit && t_entry > t_entry_prev ) {
		t_entry_prev = t_entry;

		// Down traversal - Working our way down to a leaf node.
		glm::vec3 p_entry = ray_o + ( t_entry * ray_dir );
		while ( !curr_node.is_leaf_node ) {
			curr_node = curr_node.isPointToLeftOfSplittingPlane( p_entry ) ? tree_nodes[curr_node.left_child_index] : tree_nodes[curr_node.right_child_index];
		}

		// We've reached a leaf node.
		// Check intersection with triangles contained in current leaf node.
        for ( int i = curr_node.first_tri_index; i < ( curr_node.first_tri_index + curr_node.num_tris ); ++i ) {
			glm::vec3 tri = tris[kd_tri_index_list[i]];
			glm::vec3 v0 = verts[( int )tri[0]];
			glm::vec3 v1 = verts[( int )tri[1]];
			glm::vec3 v2 = verts[( int )tri[2]];

			// Perform ray/triangle intersection test.
			float tmp_t = INFINITY;
			glm::vec3 tmp_normal( 0.0f, 0.0f, 0.0f );
			bool intersects_tri = Intersections::triIntersect( ray_o, ray_dir, v0, v1, v2, tmp_t, tmp_normal );

			if ( intersects_tri ) {
				if ( tmp_t < t_exit ) {
					intersection_detected = true;
					t_exit = tmp_t;
					normal = tmp_normal;
				}
			}
		}

		// Compute distance along ray to exit current node.
		float tmp_t_near, tmp_t_far;
		bool intersects_curr_node_bounding_box = Intersections::aabbIntersect( curr_node.bbox, ray_o, ray_dir, tmp_t_near, tmp_t_far );
		if ( intersects_curr_node_bounding_box ) {
			// Set t_entry to be the entrance point of the next (neighboring) node.
			t_entry = tmp_t_far;
		}
		else {
			// This should never happen.
			// If it does, then that means we're checking triangles in a node that the ray never intersects.
			break;
		}

		// Get neighboring node using ropes attached to current node.
		glm::vec3 p_exit = ray_o + ( t_entry * ray_dir );
		int new_node_index = curr_node.getNeighboringNodeIndex( p_exit );

		// Break if neighboring node not found, meaning we've exited the kd-tree.
		if ( new_node_index = -1 ) {
			break;
		}

		curr_node = tree_nodes[new_node_index];
	}

	if ( intersection_detected ) {
		t = t_exit;
		hit_point = ray_o + ( t * ray_dir );
		return true;
	}

	return false;
}