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
	//for (int i=0; i<nodes.size(); i+=1){
	//	delete nodes[i];
	//}
	for (int i=0; i<tris.size(); i+=1){
		delete tris[i];
	}
}


void KDTreeCPU::build( mesh *m )
{
	//// Populate list of vertices.
	//for ( int i = 0; i < m->numVerts; ++i ) {
	//	verts.push_back( m->verts[i] );
	//}
	//verts_xsorted = verts;
	//verts_ysorted = verts;
	//verts_zsorted = verts;
	//
	//// Sort vertices on x-, y-, and z-coordinates.
	//std::sort( verts_xsorted.begin(), verts_xsorted.end(), utilityCore::lessThanVec3X() );
	//std::sort( verts_ysorted.begin(), verts_ysorted.end(), utilityCore::lessThanVec3Y() );
	//std::sort( verts_zsorted.begin(), verts_zsorted.end(), utilityCore::lessThanVec3Z() );

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

	// Base case.
	if ( triangles.size() <= NUM_TRIS_PER_NODE ) {
		return node;
	}

	// Get longest side of bounding box.
	Axis longest_side = node->bbox.getLongestSide();

	if ( longest_side == XAXIS ) {
		std::sort( triangles.begin(), triangles.end(), utilityCore::lessThanTriX() );
	}
	else if ( longest_side == YAXIS ) {
		std::sort( triangles.begin(), triangles.end(), utilityCore::lessThanTriY() );
	}
	else {
		std::sort( triangles.begin(), triangles.end(), utilityCore::lessThanTriZ() );
	}

	std::vector<Triangle*>::const_iterator it_front = triangles.begin();
	std::vector<Triangle*>::const_iterator it_mid = triangles.begin() + ( triangles.size() / 2 );
	std::vector<Triangle*>::const_iterator it_back = triangles.end();

	// Create left and right trees of triangles.
	std::vector<Triangle*> left_triangles( it_front, it_mid );
	std::vector<Triangle*> right_triangles( it_mid + 1, it_back );

	// Recurse.
	node->left = build( left_triangles, depth + 1 );
	node->right = build( right_triangles, depth + 1 );

	return node;
}

KDTreeNode* KDTreeCPU::getRootNode()
{
	return root;
}