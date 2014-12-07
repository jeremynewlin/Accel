#include "KDTreeStructs.h"
#include <iostream>


////////////////////////////////////////////////////
// KDTreeNode.
////////////////////////////////////////////////////

KDTreeNode::KDTreeNode()
{
	left = NULL;
	right = NULL;
	is_leaf_node = false;
	for ( int i = 0; i < 6; ++i ) {
		ropes[i] = NULL;
	}
}

KDTreeNode::~KDTreeNode()
{
	if ( num_tris > 0 ) {
		delete[] tri_indices;
	}

	if ( left ) {
		delete left;
	}
	if ( right ) {
		delete right;
	}
}

bool KDTreeNode::isPointToLeftOfSplittingPlane( const glm::vec3 &p ) const
{
	if ( split_plane_axis == X_AXIS ) {
		return ( p.x < split_plane_value );
	}
	else if ( split_plane_axis == Y_AXIS ) {
		return ( p.y < split_plane_value );
	}
	else if ( split_plane_axis == Z_AXIS ) {
		return ( p.z < split_plane_value );
	}
	// Something went wrong because split_plane_axis is not set to one of the three allowed values.
	else {
		std::cout << "ERROR: split_plane_axis not set to valid value." << std::endl;
		std::cin.ignore();
		return false;
	}
}

KDTreeNode* KDTreeNode::getNeighboringNode( glm::vec3 p )
{
	// Check left face.
	if ( fabs( p.x - bbox.min.x ) < KD_TREE_EPSILON ) {
		return ropes[LEFT];     
	}
	// Check front face.
	else if ( fabs( p.z - bbox.max.z ) < KD_TREE_EPSILON ) {
		return ropes[FRONT];
	}
	// Check right face.
	else if ( fabs( p.x - bbox.max.x ) < KD_TREE_EPSILON ) {
		return ropes[RIGHT];
	}
	// Check back face.
	else if ( fabs( p.z - bbox.min.z ) < KD_TREE_EPSILON ) {
		return ropes[BACK];
	}
	// Check top face.
	else if ( fabs( p.y - bbox.max.y ) < KD_TREE_EPSILON ) {
		return ropes[TOP];
	}
	// Check bottom face.
	else if ( fabs( p.y - bbox.min.y ) < KD_TREE_EPSILON ) {
		return ropes[BOTTOM];
	}
	// p should be a point on one of the faces of this node's bounding box, but in this case, it isn't.
	else {
		std::cout << "ERROR: Node neighbor could not be returned." << std::endl;
		//std::cin.ignore();
		return NULL;
	}
}

void KDTreeNode::prettyPrint()
{
	std::cout << "id: " << id << std::endl;
	std::cout << "bounding box min: ( " << bbox.min.x << ", " << bbox.min.y << ", " << bbox.min.z << " )" << std::endl;
	std::cout << "bounding box max: ( " << bbox.max.x << ", " << bbox.max.y << ", " << bbox.max.z << " )" << std::endl;
	std::cout << "num_tris: " << num_tris << std::endl;

	// Print triangle indices.
	int num_tris_to_print = ( num_tris > 10 ) ? 10 : num_tris;
	for ( int i = 0; i < num_tris_to_print; ++i ) {
		std::cout << "tri index " << i << ": " << tri_indices[i] << std::endl;
	}

	// Print split plane axis.
	if ( split_plane_axis == X_AXIS ) {
		std::cout << "split plane axis: X_AXIS" << std::endl;
	}
	else if ( split_plane_axis == Y_AXIS ) {
		std::cout << "split plane axis: Y_AXIS" << std::endl;
	}
	else if ( split_plane_axis == Z_AXIS ) {
		std::cout << "split plane axis: Z_AXIS" << std::endl;
	}
	else {
		std::cout << "split plane axis: invalid" << std::endl;
	}

	std::cout << "split plane value: " << split_plane_value << std::endl;

	// Print whether or not node is a leaf node.
	if ( is_leaf_node ) {
		std::cout << "is leaf node: YES" << std::endl;
	}
	else {
		std::cout << "is leaf node: NO" << std::endl;
	}

	// Print pointers to children.
	if ( left ) {
		std::cout << "left child: " << left << std::endl;
	}
	else {
		std::cout << "left child: NULL" << std::endl;
	}
	if ( right ) {
		std::cout << "right child: " << right << std::endl;
	}
	else {
		std::cout << "right child: NULL" << std::endl;
	}

	// Print neighboring nodes.
	for ( int i = 0; i < 6; ++i ) {
		if ( ropes[i] ) {
			std::cout << "rope " << i << ": " << ropes[i] << std::endl;
		}
		else {
			std::cout << "rope " << i << ": NULL" << std::endl;
		}
	}
}


////////////////////////////////////////////////////
// KDTreeNodeGPU.
////////////////////////////////////////////////////

KDTreeNodeGPU::KDTreeNodeGPU()
{
	left_child_index = -1;
	right_child_index = -1;
	first_tri_index = -1;
	num_tris = 0;

	for ( int i = 0; i < 6; ++i ) {
		neighbor_node_indices[i] = -1;
	}
}

void KDTreeNodeGPU::prettyPrint()
{
	std::cout << "bounding box min: ( " << bbox.min.x << ", " << bbox.min.y << ", " << bbox.min.z << " )" << std::endl;
	std::cout << "bounding box max: ( " << bbox.max.x << ", " << bbox.max.y << ", " << bbox.max.z << " )" << std::endl;
	std::cout << "num_tris: " << num_tris << std::endl;
	std::cout << "first_tri_index: " << first_tri_index << std::endl;

	// Print split plane axis.
	if ( split_plane_axis == X_AXIS ) {
		std::cout << "split plane axis: X_AXIS" << std::endl;
	}
	else if ( split_plane_axis == Y_AXIS ) {
		std::cout << "split plane axis: Y_AXIS" << std::endl;
	}
	else if ( split_plane_axis == Z_AXIS ) {
		std::cout << "split plane axis: Z_AXIS" << std::endl;
	}
	else {
		std::cout << "split plane axis: invalid" << std::endl;
	}

	std::cout << "split plane value: " << split_plane_value << std::endl;

	// Print children indices.
	std::cout << "left child index: " << left_child_index << std::endl;
	std::cout << "right child index: " << right_child_index << std::endl;

	// Print neighboring nodes.
	for ( int i = 0; i < 6; ++i ) {
		if ( neighbor_node_indices[i] ) {
			std::cout << "neighbor node index " << i << ": " << neighbor_node_indices[i] << std::endl;
		}
		else {
			std::cout << "neighbor node index " << i << ": NULL" << std::endl;
		}
	}
}