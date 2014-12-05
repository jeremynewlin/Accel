#ifndef KD_TREE_CPU_H
#define KD_TREE_CPU_H

#include "boundingBox.h"
#include "utils.h"
#include <limits>


////////////////////////////////////////////////////
// Constants.
////////////////////////////////////////////////////

#define XAXIS 0
#define YAXIS 1
#define ZAXIS 2

const int NUM_TRIS_PER_NODE = 20;
const bool USE_TIGHT_FITTING_BOUNDING_BOXES = false;
const float INFINITY = std::numeric_limits<float>::max();


////////////////////////////////////////////////////
// KDTreeNode.
////////////////////////////////////////////////////
class KDTreeNode
{
public:
	KDTreeNode( void )
	{
		left = NULL;
		right = NULL;
	}

	~KDTreeNode( void )
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

	boundingBox bbox;
	KDTreeNode *left;
	KDTreeNode *right;
	int num_tris;
	int *tri_indices;
};


////////////////////////////////////////////////////
// KDTreeCPU.
////////////////////////////////////////////////////
class KDTreeCPU
{
public:
	KDTreeCPU( int num_tris, glm::vec3 *tris, int num_verts, glm::vec3 *verts );
	~KDTreeCPU( void );

	// Public traversal method that begins recursive search.
	bool intersect( const glm::vec3 &ray_o, const glm::vec3 &ray_dir, float &t, glm::vec3 &hit_point, glm::vec3 &normal ) const;

	// kd-tree getters.
	KDTreeNode* getRootNode( void ) const;
	int getNumLevels( void ) const;
	int getNumLeaves( void ) const;

	// Debug methods.
	void printNumTrianglesInEachNode( KDTreeNode *curr_node, int curr_depth = 1 );

private:
	// kd-tree variables.
	KDTreeNode *root;
	int num_levels, num_leaves;

	// Input mesh variables.
	int num_verts, num_tris;
	glm::vec3 *verts, *tris;

	KDTreeNode* constructTreeMedianSpaceSplit( int num_tris, int *tri_indices, boundingBox bounds, int curr_depth );

	// Private recursive traversal method.
	bool intersect( KDTreeNode *curr_node, const glm::vec3 &ray_o, const glm::vec3 &ray_dir, float &t, glm::vec3 &normal ) const;

	// Bounding box getters.
	int getLongestBoundingBoxSide( glm::vec3 min, glm::vec3 max );
	boundingBox computeTightFittingBoundingBox( int num_verts, glm::vec3 *verts );
	boundingBox computeTightFittingBoundingBox( int num_tris, int *tri_indices );

	// Triangle getters.
	float getMinTriValue( int tri_index, int axis );
	float getMaxTriValue( int tri_index, int axis );
};

#endif