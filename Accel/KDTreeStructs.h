#ifndef KD_TREE_STRUCTS_H
#define KD_TREE_STRUCTS_H

#include "boundingBox.h"


////////////////////////////////////////////////////
// const.
////////////////////////////////////////////////////

const float KD_TREE_EPSILON = pow( 1.0f, -5 );


////////////////////////////////////////////////////
// enums.
////////////////////////////////////////////////////

enum SplitAxis {
	X_AXIS = 0,
	Y_AXIS = 1,
	Z_AXIS = 2
};

enum AABBFace {
	LEFT = 0,
	FRONT = 1,
	RIGHT = 2,
	BACK = 3,
	TOP = 4,
	BOTTOM = 5
};


////////////////////////////////////////////////////
// structs.
////////////////////////////////////////////////////

struct Ray
{
	glm::vec3 origin;
	glm::vec3 dir;
};


////////////////////////////////////////////////////
// classes.
////////////////////////////////////////////////////

class KDTreeNode
{
public:
	KDTreeNode( void );
	~KDTreeNode( void );

	boundingBox bbox;
	KDTreeNode *left;
	KDTreeNode *right;
	int num_tris;
	int *tri_indices;

	SplitAxis split_plane_axis;
	float split_plane_value;

	bool is_leaf_node;

	// One rope for each face of the AABB encompassing the triangles in a node.
	KDTreeNode *ropes[6];

	bool isPointToLeftOfSplittingPlane( const glm::vec3 &p ) const;
	KDTreeNode* getNeighboringNode( glm::vec3 p );
};

#endif