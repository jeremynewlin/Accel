#ifndef KD_TREE_CPU_H
#define KD_TREE_CPU_H

#include "Triangle.h"
#include "boundingBox.h"
#include "mesh.h"
#include "utils.h"
#include <vector>


const int NUM_TRIS_PER_NODE = 50;
const int MAX_NUM_LEVELS = 20;
const bool USE_TIGHT_FITTING_BOUNDING_BOXES = true;


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
		for ( int i = 0; i < tris.size(); ++i ){
			delete tris[i];
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
	std::vector<Triangle*> tris;
	Axis split_plane_axis;
};


class KDTreeCPU
{
public:
	KDTreeCPU( int num_tris, glm::vec3 *tris, int num_verts, glm::vec3 *verts );
	~KDTreeCPU( void );

	KDTreeNode* getRootNode( void );
	int getMaxNumLevels( void );

private:
	KDTreeNode *root;
	int max_num_levels;

	std::vector<Triangle*> mesh_tris;

	KDTreeNode* constructTreeMedianSpaceSplit( std::vector<Triangle*> tri_list, boundingBox bounds, int curr_depth );
	KDTreeNode* constructTreeMedianVertexSplit( std::vector<Triangle*> tri_list, boundingBox bounds, int curr_depth );
	KDTreeNode* constructTreeMedianTriangleCentroidSplit( std::vector<Triangle*> tri_list, boundingBox bounds, int curr_depth );

	std::vector<glm::vec3> getVertListFromTriList( std::vector<Triangle*> tri_list );
};

#endif