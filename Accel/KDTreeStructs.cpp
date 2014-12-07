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