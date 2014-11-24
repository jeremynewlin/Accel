#include "KDTreeCPU.h"
#include <algorithm>


KDTreeCPU::KDTreeCPU()
{
	root = NULL;
}

KDTreeCPU::KDTreeCPU( mesh *m )
{
	build( m );
}


KDTreeCPU::~KDTreeCPU( void )
{
	// TODO: Cleanup.
	for (int i=0; i<nodes.size(); i+=1){
		delete nodes[i];
	}
	for (int i=0; i<tris.size(); i+=1){
		delete tris[i];
	}
}


void KDTreeCPU::build( mesh *m )
{
	// Populate list of vertices.
	for ( int i = 0; i < m->numVerts; ++i ) {
		verts.push_back( m->verts[i] );
	}
	verts_xsorted = verts;
	verts_ysorted = verts;
	verts_zsorted = verts;
	
	// Sort vertices on x-, y-, and z-coordinates.
	std::sort( verts_xsorted.begin(), verts_xsorted.end(), utilityCore::lessThanVec3X() );
	std::sort( verts_ysorted.begin(), verts_ysorted.end(), utilityCore::lessThanVec3Y() );
	std::sort( verts_zsorted.begin(), verts_zsorted.end(), utilityCore::lessThanVec3Z() );

	// TODO: When sorting object vertices, vertex indices of mesh triangles must be updated as well.
	// These sorted lists can be used to implement a less naive kd-tree construction method.

	// Populate list of triangle objects.
	for ( int i = 0; i < m->numTris; ++i ) {
		glm::vec3 face = m->tris[i];
		glm::vec3 v1 = m->verts[(int)face[0]];
		glm::vec3 v2 = m->verts[(int)face[1]];
		glm::vec3 v3 = m->verts[(int)face[2]];
		tris.push_back( new Triangle( v1, v2, v3 ) );
	}

	root = build( tris, 0 );
}


KDTreeNode* KDTreeCPU::build( std::vector<Triangle*> triangles, int depth )
{
	KDTreeNode *node = new KDTreeNode();
	node->tris = triangles;

	// Get bounding box that emcompasses all triangles in node.
	node->bbox = boundingBox( triangles );

	nodes.push_back(node);

	// Base case.
	if ( triangles.size() <= NUM_TRIS_PER_NODE ) {
		return node;
	}

	// Compute midpoint of all triangle vertices.
	glm::vec3 mid( 0.0f, 0.0f, 0.0f );
	for ( std::vector<Triangle*>::iterator it = triangles.begin(); it != triangles.end(); ++it ) {
		Triangle *tri = *it;
		mid += tri->center;
	}
	mid /= (float)triangles.size();

	// Create left and right trees of triangles.
	std::vector<Triangle*> left_triangles;
	std::vector<Triangle*> right_triangles;
	for ( std::vector<Triangle*>::iterator it = triangles.begin(); it != triangles.end(); ++it ) {
		Triangle *tri = *it;

		if ( depth % 3 == 0 ) {
			mid.x >= tri->center.x ? right_triangles.push_back( tri ) : left_triangles.push_back( tri );
		}
		else if ( depth % 3 == 1 ) {
			mid.y >= tri->center.y ? right_triangles.push_back( tri ) : left_triangles.push_back( tri );
		}
		else {
			mid.z >= tri->center.z ? right_triangles.push_back( tri ) : left_triangles.push_back( tri );
		}
	}

	// Recurse.
	node->left = build( left_triangles, depth + 1 );
	node->right = build( right_triangles, depth + 1 );

	return node;
}

boundingBox KDTreeCPU::getBoundingBox(int i){
	return nodes[i]->bbox;
}

int KDTreeCPU::getNumNodes(){
	return nodes.size();
}