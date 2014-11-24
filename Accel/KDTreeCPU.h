#ifndef KD_TREE_CPU_H
#define KD_TREE_CPU_H

#include "Triangle.h"
#include "boundingBox.h"
#include "mesh.h"
#include "utils.h"
#include <vector>


const int NUM_TRIS_PER_NODE = 20;


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
	}

	boundingBox bbox;
	KDTreeNode *left;
	KDTreeNode *right;
	std::vector<Triangle*> tris;
};


class KDTreeCPU
{
public:
	KDTreeCPU( void );
	KDTreeCPU( mesh *m );

	~KDTreeCPU( void );

	void build( mesh *m );

	boundingBox getBoundingBox(int i);
	int getNumNodes();

private:
	KDTreeNode *root;

	std::vector<glm::vec3> verts;
	std::vector<glm::vec3> verts_xsorted, verts_ysorted, verts_zsorted;
	std::vector<Triangle*> tris;

	std::vector<KDTreeNode*> nodes;

	KDTreeNode* build( std::vector<Triangle*> triangles, int depth );
};

#endif