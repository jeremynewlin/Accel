#include "KDTreeCPU.h"
#include <algorithm>


////////////////////////////////////////////////////
// Constructor/destructor.
////////////////////////////////////////////////////

KDTreeCPU::KDTreeCPU( int num_tris, glm::vec3 *tris, int num_verts, glm::vec3 *verts )
{
	max_num_levels = 0;

	// Populate list of triangle objects.
	for ( int i = 0; i < num_tris; ++i ) {
		glm::vec3 face = tris[i];
		glm::vec3 v1 = verts[( int )face[0]];
		glm::vec3 v2 = verts[( int )face[1]];
		glm::vec3 v3 = verts[( int )face[2]];
		mesh_tris.push_back( new Triangle( v1, v2, v3 ) );
	}

	// Build kd-tree and set root node.
	root = constructTreeMedianSpaceSplit( mesh_tris, boundingBox( mesh_tris ), 1 );
	//root = constructTreeMedianTriangleCentroidSplit( mesh_tris, boundingBox( mesh_tris ), 1 );
	//root = constructTreeMedianVertexSplit( mesh_tris, boundingBox( mesh_tris ), 1 );
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

KDTreeNode* KDTreeCPU::getRootNode() const
{
	return root;
}

int KDTreeCPU::getMaxNumLevels() const
{
	return max_num_levels;
}


////////////////////////////////////////////////////
// intersect().
// o: Ray origin. dir: Ray direction.
////////////////////////////////////////////////////
bool KDTreeCPU::intersect( KDTreeNode *node, glm::vec3 o, glm::vec3 dir ) const
{
	// TODO: If ray intersects node->bbox.
	if ( false ) {
		if ( node->left || node->right ) {
			bool hit_left = intersect( node->left, o, dir );
			bool hit_right = intersect( node->right, o, dir );
			return hit_left || hit_right;
		}
		else {
			// Leaf node.
			for ( int i = 0; i < node->tris.size(); ++i ) {
				// TODO: If ray intersects current triangle.
				if ( false ) {
					// Record closest intersection point information.
				}
			}

			// TODO: If triangle intersection was detected.
			if ( false ) {
				// TODO: Set intersection information caller might want to know about.
				return true;
			}
		}
	}

	return false;
}


////////////////////////////////////////////////////
// constructTreeMedianSpaceSplit().
////////////////////////////////////////////////////
KDTreeNode* KDTreeCPU::constructTreeMedianSpaceSplit( std::vector<Triangle*> tri_list, boundingBox bounds, int curr_depth )
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
		if ( curr_depth > max_num_levels ) {
			max_num_levels = curr_depth;
		}
		return node;
	}

	// Get longest side of bounding box.
	Axis longest_side = node->bbox.getLongestSide();

	// Set split plane for node.
	node->split_plane_axis = longest_side;

	// Define "loose-fitting" bounding boxes.
	boundingBox left_bbox = bounds;
	boundingBox right_bbox = bounds;

	// Define split plane value.
	float median_val = 0.0;

	// Sort list of vertices and compute "loose-fitting" bounding boxes.
	if ( longest_side == XAXIS ) {
		median_val = bounds.min.x + ( ( bounds.max.x - bounds.min.x ) / 2.0f );
		left_bbox.max.x = median_val;
		right_bbox.min.x = median_val;
	}
	else if ( longest_side == YAXIS ) {
		median_val = bounds.min.y + ( ( bounds.max.y - bounds.min.y ) / 2.0f );
		left_bbox.max.y = median_val;
		right_bbox.min.y = median_val;
	}
	else {
		median_val = bounds.min.z + ( ( bounds.max.z - bounds.min.z ) / 2.0f );
		left_bbox.max.z = median_val;
		right_bbox.min.z = median_val;
	}

	// Split list of triangles into left and right subtrees.
	std::vector<Triangle*> left_tris;
	std::vector<Triangle*> right_tris;
	for ( int i = 0; i < tri_list.size(); ++i ) {
		glm::vec3 tri_min = tri_list[i]->getMin();
		glm::vec3 tri_max = tri_list[i]->getMax();

		if ( longest_side == XAXIS ) {
			if ( tri_min.x < median_val ) {
				left_tris.push_back( tri_list[i] );
			}
			if ( tri_max.x >= median_val ) {
				right_tris.push_back( tri_list[i] );
			}
		}
		else if ( longest_side == YAXIS ) {
			if ( tri_min.y < median_val ) {
				left_tris.push_back( tri_list[i] );
			}
			if ( tri_max.y >= median_val ) {
				right_tris.push_back( tri_list[i] );
			}
		}
		else {
			if ( tri_min.z < median_val ) {
				left_tris.push_back( tri_list[i] );
			}
			if ( tri_max.z >= median_val ) {
				right_tris.push_back( tri_list[i] );
			}
		}
	}

	// Recurse.
	node->left = constructTreeMedianVertexSplit( left_tris, left_bbox, curr_depth + 1 );
	node->right = constructTreeMedianVertexSplit( right_tris, right_bbox, curr_depth + 1 );

	return node;
}


////////////////////////////////////////////////////
// constructTreeMedianTriangleCentroidSplit().
////////////////////////////////////////////////////
KDTreeNode* KDTreeCPU::constructTreeMedianTriangleCentroidSplit( std::vector<Triangle*> tri_list, boundingBox bounds, int curr_depth )
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
		if ( curr_depth > max_num_levels ) {
			max_num_levels = curr_depth;
		}
		return node;
	}

	// Get longest side of bounding box.
	Axis longest_side = node->bbox.getLongestSide();

	// Set split plane for node.
	node->split_plane_axis = longest_side;

	// Define "loose-fitting" bounding boxes.
	boundingBox left_bbox = bounds;
	boundingBox right_bbox = bounds;

	// Sort list of vertices and compute "loose-fitting" bounding boxes.
	if ( longest_side == XAXIS ) {
		std::sort( tri_list.begin(), tri_list.end(), utilityCore::lessThanTriX() );
		left_bbox.max.x = bounds.min.x + ( ( bounds.max.x - bounds.min.x ) / 2.0f );
		right_bbox.min.x = bounds.min.x + ( ( bounds.max.x - bounds.min.x ) / 2.0f );
	}
	else if ( longest_side == YAXIS ) {
		std::sort( tri_list.begin(), tri_list.end(), utilityCore::lessThanTriY() );
		left_bbox.max.y = bounds.min.y + ( ( bounds.max.y - bounds.min.y ) / 2.0f );
		right_bbox.min.y = bounds.min.y + ( ( bounds.max.y - bounds.min.y ) / 2.0f );
	}
	else {
		std::sort( tri_list.begin(), tri_list.end(), utilityCore::lessThanTriZ() );
		left_bbox.max.z = bounds.min.z + ( ( bounds.max.z - bounds.min.z ) / 2.0f );
		right_bbox.min.z = bounds.min.z + ( ( bounds.max.z - bounds.min.z ) / 2.0f );
	}

	// Get median vetex value to split on.
	int median_tri_index = tri_list.size() / 2;
	glm::vec3 median_point = tri_list[median_tri_index]->center;

	// Split list of triangles into left and right subtrees.
	std::vector<Triangle*> left_tris;
	std::vector<Triangle*> right_tris;
	for ( int i = 0; i < tri_list.size(); ++i ) {
		glm::vec3 tri_min = tri_list[i]->getMin();
		glm::vec3 tri_max = tri_list[i]->getMax();

		if ( longest_side == XAXIS ) {
			if ( tri_min.x < median_point.x ) {
				left_tris.push_back( tri_list[i] );
			}
			if ( tri_max.x >= median_point.x ) {
				right_tris.push_back( tri_list[i] );
			}
		}
		else if ( longest_side == YAXIS ) {
			if ( tri_min.y < median_point.y ) {
				left_tris.push_back( tri_list[i] );
			}
			if ( tri_max.y >= median_point.y ) {
				right_tris.push_back( tri_list[i] );
			}
		}
		else {
			if ( tri_min.z < median_point.z ) {
				left_tris.push_back( tri_list[i] );
			}
			if ( tri_max.z >= median_point.z ) {
				right_tris.push_back( tri_list[i] );
			}
		}
	}

	// Recurse.
	node->left = constructTreeMedianVertexSplit( left_tris, left_bbox, curr_depth + 1 );
	node->right = constructTreeMedianVertexSplit( right_tris, right_bbox, curr_depth + 1 );

	return node;
}


////////////////////////////////////////////////////
// getVertListFromTriList().
////////////////////////////////////////////////////
std::vector<glm::vec3> KDTreeCPU::getVertListFromTriList( std::vector<Triangle*> tri_list ) const
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
////////////////////////////////////////////////////
KDTreeNode* KDTreeCPU::constructTreeMedianVertexSplit( std::vector<Triangle*> tri_list, boundingBox bounds, int curr_depth )
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
		if ( curr_depth > max_num_levels ) {
			max_num_levels = curr_depth;
		}
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
	node->left = constructTreeMedianVertexSplit( left_tris, left_bbox, curr_depth + 1 );
	node->right = constructTreeMedianVertexSplit( right_tris, right_bbox, curr_depth + 1 );

	return node;
}