#include "KDTreeCPU.h"
#include <algorithm>


////////////////////////////////////////////////////
// Constructor/destructor.
////////////////////////////////////////////////////

KDTreeCPU::KDTreeCPU( int num_tris, glm::vec3 *tris, int num_verts, glm::vec3 *verts )
{
	// Populate list of triangle objects.
	for ( int i = 0; i < num_tris; ++i ) {
		glm::vec3 face = tris[i];
		glm::vec3 v1 = verts[( int )face[0]];
		glm::vec3 v2 = verts[( int )face[1]];
		glm::vec3 v3 = verts[( int )face[2]];
		mesh_tris.push_back( new Triangle( v1, v2, v3 ) );
	}

	// Build kd-tree and set root node.
	root = constructTreeMedianVertexSplit( mesh_tris, boundingBox( mesh_tris ) );
}

KDTreeCPU::~KDTreeCPU()
{
	for ( int i = 0; i < mesh_tris.size(); ++i ){
		delete mesh_tris[i];
	}

	delete root;
}


////////////////////////////////////////////////////
// Getters.
////////////////////////////////////////////////////

KDTreeNode* KDTreeCPU::getRootNode()
{
	return root;
}


////////////////////////////////////////////////////
// getVertListFromTriList().
////////////////////////////////////////////////////
std::vector<glm::vec3> KDTreeCPU::getVertListFromTriList( std::vector<Triangle*> tri_list )
{
	std::vector<glm::vec3> vert_list;
	for ( int i = 0; i < tri_list.size(); ++i ) {
		vert_list.push_back( tri_list[i]->v1 );
		vert_list.push_back( tri_list[i]->v2 );
		vert_list.push_back( tri_list[i]->v3 );
	}

	// Remove duplicates.
	std::sort( vert_list.begin(), vert_list.end(), utilityCore::lessThanVec3X() );
	vert_list.erase( std::unique( vert_list.begin(), vert_list.end() ), vert_list.end() );

	return vert_list;
}


////////////////////////////////////////////////////
// constructTreeMedianVertexSplit().
// ERROR: Calling this method results in overflowed stack.
////////////////////////////////////////////////////
KDTreeNode* KDTreeCPU::constructTreeMedianVertexSplit( std::vector<Triangle*> tri_list, boundingBox bounds )
{
	// Create new node.
	KDTreeNode *node = new KDTreeNode();
	node->tris = tri_list;

	// Override passed-in bounding box and create "tightest-fitting" bounding box around passed-in list of triangles.
	if ( USE_TIGHT_FITTING_BOUNDING_BOXES ) {
		node->bbox = boundingBox( tri_list );
	}
	else {
		node->bbox = bounds;
	}

	// Base case--Number of triangles in node is small enough.
	if ( tri_list.size() <= NUM_TRIS_PER_NODE ) {
		return node;
	}

	// Create list of vertices from passed-in list of triangles.
	std::vector<glm::vec3> vert_list = getVertListFromTriList( tri_list );

	// Get longest side of bounding box.
	Axis longest_side = node->bbox.getLongestSide();

	// Set split plane for node.
	node->split_plane_axis = longest_side;

	// Define "loose-fitting" bounding boxes.
	boundingBox left_bbox = bounds;
	boundingBox right_bbox = bounds;

	// Sort list of vertices and compute "loose-fitting" bounding boxes.
	if ( longest_side == XAXIS ) {
		std::sort( vert_list.begin(), vert_list.end(), utilityCore::lessThanVec3X() );
		left_bbox.max.x = bounds.min.x + ( ( bounds.max.x - bounds.min.x ) / 2.0f );
		right_bbox.min.x = bounds.min.x + ( ( bounds.max.x - bounds.min.x ) / 2.0f );
	}
	else if ( longest_side == YAXIS ) {
		std::sort( vert_list.begin(), vert_list.end(), utilityCore::lessThanVec3Y() );
		left_bbox.max.y = bounds.min.y + ( ( bounds.max.y - bounds.min.y ) / 2.0f );
		right_bbox.min.y = bounds.min.y + ( ( bounds.max.y - bounds.min.y ) / 2.0f );
	}
	else {
		std::sort( vert_list.begin(), vert_list.end(), utilityCore::lessThanVec3Z() );
		left_bbox.max.z = bounds.min.z + ( ( bounds.max.z - bounds.min.z ) / 2.0f );
		right_bbox.min.z = bounds.min.z + ( ( bounds.max.z - bounds.min.z ) / 2.0f );
	}

	// Get median vetex value to split on.
	int median_vert_index = vert_list.size() / 2;
	glm::vec3 median_vert = vert_list[median_vert_index];

	// Split list of triangles into left and right subtrees.
	std::vector<Triangle*> left_tris;
	std::vector<Triangle*> right_tris;
	for ( int i = 0; i < tri_list.size(); ++i ) {
		glm::vec3 tri_min = tri_list[i]->getMin();
		glm::vec3 tri_max = tri_list[i]->getMax();

		if ( longest_side == XAXIS ) {
			if ( tri_min.x < median_vert.x ) {
				left_tris.push_back( tri_list[i] );
			}
			if ( tri_max.x >= median_vert.x ) {
				right_tris.push_back( tri_list[i] );
			}
		}
		else if ( longest_side == YAXIS ) {
			if ( tri_min.y < median_vert.y ) {
				left_tris.push_back( tri_list[i] );
			}
			if ( tri_max.y >= median_vert.y ) {
				right_tris.push_back( tri_list[i] );
			}
		}
		else {
			if ( tri_min.z < median_vert.z ) {
				left_tris.push_back( tri_list[i] );
			}
			if ( tri_max.z >= median_vert.z ) {
				right_tris.push_back( tri_list[i] );
			}
		}
	}

	// Recurse.
	node->left = constructTreeMedianVertexSplit( left_tris, left_bbox );
	node->right = constructTreeMedianVertexSplit( right_tris, right_bbox );

	return node;
}