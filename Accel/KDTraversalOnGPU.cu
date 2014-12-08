#include "KDTraversalOnGPU.h"
#include <limits>


glm::vec3* cudaRayCastObj( Camera *camera, mesh *obj_mesh, KDTreeGPU *kd_tree, bool use_brute_force_approach )
{
	int camera_raycast_tile_size = 8;

	dim3 threads_per_block( camera_raycast_tile_size, camera_raycast_tile_size );
	dim3 full_blocks_per_grid( ( int )ceil( ( float )camera->getResolution().x / ( float )camera_raycast_tile_size ),
							   ( int )ceil( ( float )camera->getResolution().y / ( float )camera_raycast_tile_size ) );
  
	// Send image to GPU.
	glm::vec3 *cuda_image = NULL;
	float size_image = ( int )camera->getResolution().x * ( int )camera->getResolution().y * sizeof( glm::vec3 );
	cudaMalloc( ( void** )&cuda_image, size_image );

	// Send mesh triangles to GPU.
	glm::vec3 *cuda_mesh_tris = NULL;
	float size_mesh_tris = obj_mesh->numTris * sizeof( glm::vec3 );
	cudaMalloc( ( void** )&cuda_mesh_tris, size_mesh_tris );
	cudaMemcpy( cuda_mesh_tris, obj_mesh->tris, size_mesh_tris, cudaMemcpyHostToDevice );

	// Send mesh vertices to GPU.
	glm::vec3 *cuda_mesh_verts = NULL;
	float size_mesh_verts = obj_mesh->numVerts * sizeof( glm::vec3 );
	cudaMalloc( ( void** )&cuda_mesh_verts, size_mesh_verts );
	cudaMemcpy( cuda_mesh_verts, obj_mesh->verts, size_mesh_verts, cudaMemcpyHostToDevice );

	// Send kd-tree nodes to GPU.
	KDTreeNodeGPU *cuda_kd_tree_nodes = NULL;
	float size_kd_tree_nodes = kd_tree->getNumNodes() * sizeof( KDTreeNodeGPU );
	cudaMalloc( ( void** )&cuda_kd_tree_nodes, size_kd_tree_nodes );
	cudaMemcpy( cuda_kd_tree_nodes, kd_tree->getTreeNodes(), size_kd_tree_nodes, cudaMemcpyHostToDevice );

	std::vector<int> kd_tree_tri_indics = kd_tree->getTriIndexList();
	int *tri_index_array = new int[kd_tree_tri_indics.size()];
	for ( int i = 0; i < kd_tree_tri_indics.size(); ++i ) {
		tri_index_array[i] = kd_tree_tri_indics[i];
	}

	// Send kd-tree triangle indices to GPU.
	int *cuda_kd_tree_tri_indices = NULL;
	float size_kd_tree_tri_indices = kd_tree_tri_indics.size() * sizeof( int );
	cudaMalloc( ( void** )&cuda_kd_tree_tri_indices, size_kd_tree_nodes );
	cudaMemcpy( cuda_kd_tree_tri_indices, tri_index_array, size_kd_tree_tri_indices, cudaMemcpyHostToDevice );

	// Call ray cast kernel.
	if ( use_brute_force_approach ) {
		performBruteForceGPURaycast<<< full_blocks_per_grid, threads_per_block >>>( cuda_image,
																					camera->getResolution(), camera->getPosition(), camera->getM() , camera->getH() , camera->getV(),
																					obj_mesh->bb, obj_mesh->numTris, cuda_mesh_tris, cuda_mesh_verts );
	}
	else {
		performGPURaycast<<< full_blocks_per_grid, threads_per_block >>>( cuda_image,
																		  camera->getResolution(), camera->getPosition(), camera->getM() , camera->getH() , camera->getV(),
																		  cuda_mesh_tris, cuda_mesh_verts,
																		  kd_tree->getRootIndex(), cuda_kd_tree_nodes, cuda_kd_tree_tri_indices );
	}

	// Copy image from device to host.
	glm::vec3 *rendered_image = new glm::vec3[( int )camera->getResolution().x * ( int )camera->getResolution().y];
	cudaMemcpy( rendered_image, cuda_image, size_image, cudaMemcpyDeviceToHost );

	// Clean up allocated memory.
	cudaFree( cuda_image );
	cudaFree( cuda_mesh_tris );
	cudaFree( cuda_mesh_verts );
	cudaFree( cuda_kd_tree_nodes );
	cudaFree( cuda_kd_tree_tri_indices );
	delete[] tri_index_array;

	return rendered_image;
}


__global__
void performGPURaycast( glm::vec3 *image_buffer,
						glm::vec2 cam_reso, glm::vec3 cam_pos, glm::vec3 cam_m, glm::vec3 cam_h, glm::vec3 cam_v,
						glm::vec3 *mesh_tris, glm::vec3 *mesh_verts,
						int kd_tree_root_index, KDTreeNodeGPU *kd_tree_nodes, int *kd_tree_tri_indices )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = ( y * ( int )cam_reso.x ) + x;

	if ( index > ( cam_reso.x * cam_reso.y ) ) {
		return;
	}

	float sx = ( float )x / ( cam_reso.x - 1.0f );
	float sy = 1.0f - ( ( float )y / ( cam_reso.y - 1.0f ) );
	glm::vec3 image_point = cam_m + ( ( 2.0f * sx - 1.0f ) * cam_h ) + ( ( 2.0f * sy - 1.0f ) * cam_v );
	glm::vec3 dir = image_point - cam_pos;

	Ray r;
	r.origin = cam_pos;
	r.dir = glm::normalize( dir );

	float t;
	glm::vec3 hit_point, normal;
	bool intersects = gpuStacklessGPUIntersect( r.origin, r.dir,
												kd_tree_root_index, kd_tree_nodes, kd_tree_tri_indices,
												mesh_tris, mesh_verts,
												t, hit_point, normal );

	glm::vec3 pixel_color( 0.0f, 0.0f, 0.0f );
	if ( intersects ) {
		pixel_color.x = ( normal.x < 0.0f ) ? ( normal.x * -1.0f ) : normal.x;
		pixel_color.y = ( normal.y < 0.0f ) ? ( normal.y * -1.0f ) : normal.y;
		pixel_color.z = ( normal.z < 0.0f ) ? ( normal.z * -1.0f ) : normal.z;
	}

	// DEBUG - Render ray directions emitted from camera.
	//glm::vec3 pixel_color = r.dir;
	//pixel_color.x = ( pixel_color.x < 0.0f ) ? ( pixel_color.x * -1.0f ) : pixel_color.x;
	//pixel_color.y = ( pixel_color.y < 0.0f ) ? ( pixel_color.y * -1.0f ) : pixel_color.y;
	//pixel_color.z = ( pixel_color.z < 0.0f ) ? ( pixel_color.z * -1.0f ) : pixel_color.z;

	image_buffer[index] = pixel_color;
	return;
}


__global__
void performBruteForceGPURaycast( glm::vec3 *image_buffer,
								  glm::vec2 cam_reso, glm::vec3 cam_pos, glm::vec3 cam_m, glm::vec3 cam_h, glm::vec3 cam_v,
								  boundingBox bbox, int num_tris, glm::vec3 *mesh_tris, glm::vec3 *mesh_verts )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = ( y * ( int )cam_reso.x ) + x;

	if ( index > ( cam_reso.x * cam_reso.y ) ) {
		return;
	}

	float sx = ( float )x / ( cam_reso.x - 1.0f );
	float sy = 1.0f - ( ( float )y / ( cam_reso.y - 1.0f ) );
	glm::vec3 image_point = cam_m + ( ( 2.0f * sx - 1.0f ) * cam_h ) + ( ( 2.0f * sy - 1.0f ) * cam_v );
	glm::vec3 dir = image_point - cam_pos;

	Ray r;
	r.origin = cam_pos;
	r.dir = glm::normalize( dir );

	glm::vec3 pixel_color( 0.0f, 0.0f, 0.0f );

	// Perform ray/AABB intersection test.
	float t_near, t_far;
	bool intersects_aabb = gpuAABBIntersect( bbox, r.origin, r.dir, t_near, t_far );

	if ( intersects_aabb ) {
		float t = 999999999;

		for ( int i = 0; i < num_tris; ++i ) {
			glm::vec3 tri = mesh_tris[i];
			glm::vec3 v0 = mesh_verts[( int )tri[0]];
			glm::vec3 v1 = mesh_verts[( int )tri[1]];
			glm::vec3 v2 = mesh_verts[( int )tri[2]];

			// Perform ray/triangle intersection test.
			float tmp_t = 999999999;
			glm::vec3 tmp_normal( 0.0f, 0.0f, 0.0f );
			bool intersects_tri = gpuTriIntersect( r.origin, r.dir, v0, v1, v2, tmp_t, tmp_normal );

			if ( intersects_tri ) {
				if ( tmp_t < t ) {
					t = tmp_t;
					pixel_color.x = ( tmp_normal.x < 0.0f ) ? ( tmp_normal.x * -1.0f ) : tmp_normal.x;
					pixel_color.y = ( tmp_normal.y < 0.0f ) ? ( tmp_normal.y * -1.0f ) : tmp_normal.y;
					pixel_color.z = ( tmp_normal.z < 0.0f ) ? ( tmp_normal.z * -1.0f ) : tmp_normal.z;

				}
			}
		}
	}

	image_buffer[index] = pixel_color;
	return;
}






////////////////////////////////////////////////////
// GPU stackless kd-tree traversal method to be called from CUDA kernel.
////////////////////////////////////////////////////
__device__
bool gpuStacklessGPUIntersect( const glm::vec3 &ray_o, const glm::vec3 &ray_dir,
							   int root_index, KDTreeNodeGPU *tree_nodes, int *kd_tri_index_list,
							   glm::vec3 *tris, glm::vec3 *verts,
							   float &t, glm::vec3 &hit_point, glm::vec3 &normal )
{
	const float GPU_INFINITY = 999999999.0f;

    KDTreeNodeGPU curr_node = tree_nodes[root_index];

	// Perform ray/AABB intersection test.
	float t_entry, t_exit;
	bool intersects_root_node_bounding_box = gpuAABBIntersect( curr_node.bbox, ray_o, ray_dir, t_entry, t_exit );

	if ( !intersects_root_node_bounding_box ) {
		return false;
	}

	bool intersection_detected = false;

	float t_entry_prev = -GPU_INFINITY;
	while ( t_entry < t_exit && t_entry > t_entry_prev ) {
		t_entry_prev = t_entry;

		// Down traversal - Working our way down to a leaf node.
		glm::vec3 p_entry = ray_o + ( t_entry * ray_dir );
		while ( !curr_node.is_leaf_node ) {
			curr_node = gpuIsPointToLeftOfSplittingPlane( curr_node, p_entry ) ? tree_nodes[curr_node.left_child_index] : tree_nodes[curr_node.right_child_index];
		}

		// We've reached a leaf node.
		// Check intersection with triangles contained in current leaf node.
        for ( int i = curr_node.first_tri_index; i < ( curr_node.first_tri_index + curr_node.num_tris ); ++i ) {
			glm::vec3 tri = tris[kd_tri_index_list[i]];
			glm::vec3 v0 = verts[( int )tri[0]];
			glm::vec3 v1 = verts[( int )tri[1]];
			glm::vec3 v2 = verts[( int )tri[2]];

			// Perform ray/triangle intersection test.
			float tmp_t = GPU_INFINITY;
			glm::vec3 tmp_normal( 0.0f, 0.0f, 0.0f );
			bool intersects_tri = gpuTriIntersect( ray_o, ray_dir, v0, v1, v2, tmp_t, tmp_normal );

			if ( intersects_tri ) {
				if ( tmp_t < t_exit ) {
					intersection_detected = true;
					t_exit = tmp_t;
					normal = tmp_normal;
				}
			}
		}

		// Compute distance along ray to exit current node.
		float tmp_t_near, tmp_t_far;
		bool intersects_curr_node_bounding_box = gpuAABBIntersect( curr_node.bbox, ray_o, ray_dir, tmp_t_near, tmp_t_far );
		if ( intersects_curr_node_bounding_box ) {
			// Set t_entry to be the entrance point of the next (neighboring) node.
			t_entry = tmp_t_far;
		}
		else {
			// This should never happen.
			// If it does, then that means we're checking triangles in a node that the ray never intersects.
			break;
		}

		// Get neighboring node using ropes attached to current node.
		glm::vec3 p_exit = ray_o + ( t_entry * ray_dir );
		int new_node_index = gpuGetNeighboringNodeIndex( curr_node, p_exit );

		// Break if neighboring node not found, meaning we've exited the kd-tree.
		if ( new_node_index == -1 ) {
			break;
		}

		curr_node = tree_nodes[new_node_index];
	}

	if ( intersection_detected ) {
		t = t_exit;
		hit_point = ray_o + ( t * ray_dir );
		return true;
	}

	return false;
}


__device__
bool gpuIsPointToLeftOfSplittingPlane( KDTreeNodeGPU node, const glm::vec3 &p )
{
	if ( node.split_plane_axis == X_AXIS ) {
		return ( p.x < node.split_plane_value );
	}
	else if ( node.split_plane_axis == Y_AXIS ) {
		return ( p.y < node.split_plane_value );
	}
	else if ( node.split_plane_axis == Z_AXIS ) {
		return ( p.z < node.split_plane_value );
	}
	// Something went wrong because split_plane_axis is not set to one of the three allowed values.
	else {
		return false;
	}
}

__device__
int gpuGetNeighboringNodeIndex( KDTreeNodeGPU node, glm::vec3 p )
{
	const float GPU_KD_TREE_EPSILON = 0.00001f;

	// Check left face.
	if ( fabs( p.x - node.bbox.min.x ) < GPU_KD_TREE_EPSILON ) {
		return node.neighbor_node_indices[LEFT];     
	}
	// Check front face.
	else if ( fabs( p.z - node.bbox.max.z ) < GPU_KD_TREE_EPSILON ) {
		return node.neighbor_node_indices[FRONT];
	}
	// Check right face.
	else if ( fabs( p.x - node.bbox.max.x ) < GPU_KD_TREE_EPSILON ) {
		return node.neighbor_node_indices[RIGHT];
	}
	// Check back face.
	else if ( fabs( p.z - node.bbox.min.z ) < GPU_KD_TREE_EPSILON ) {
		return node.neighbor_node_indices[BACK];
	}
	// Check top face.
	else if ( fabs( p.y - node.bbox.max.y ) < GPU_KD_TREE_EPSILON ) {
		return node.neighbor_node_indices[TOP];
	}
	// Check bottom face.
	else if ( fabs( p.y - node.bbox.min.y ) < GPU_KD_TREE_EPSILON ) {
		return node.neighbor_node_indices[BOTTOM];
	}
	// p should be a point on one of the faces of this node's bounding box, but in this case, it isn't.
	else {
		return -1;
	}
}


__device__
float findMax( const float &a, const float &b )
{
	return ( a >= b ) ? a : b;
}


__device__ float findMin( const float &a, const float &b )
{
	return ( a <= b ) ? a : b;
}


////////////////////////////////////////////////////
// Fast ray/AABB intersection test.
// Implementation inspired by zacharmarz.
// https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
////////////////////////////////////////////////////
__device__
bool gpuAABBIntersect( boundingBox bbox, glm::vec3 ray_o, glm::vec3 ray_dir, float &t_near, float &t_far )
{
	glm::vec3 dirfrac( 1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z );

	float t1 = ( bbox.min.x - ray_o.x ) * dirfrac.x;
	float t2 = ( bbox.max.x - ray_o.x ) * dirfrac.x;
	float t3 = ( bbox.min.y - ray_o.y ) * dirfrac.y;
	float t4 = ( bbox.max.y - ray_o.y ) * dirfrac.y;
	float t5 = ( bbox.min.z - ray_o.z ) * dirfrac.z;
	float t6 = ( bbox.max.z - ray_o.z ) * dirfrac.z;

	float tmin = findMax( findMax( findMin( t1, t2 ), findMin( t3, t4 ) ), findMin( t5, t6 ) );
	float tmax = findMin( findMin( findMax( t1, t2 ), findMax( t3, t4 ) ), findMax( t5, t6 ) );

	// If tmax < 0, ray intersects AABB, but entire AABB is behind ray, so reject.
	if ( tmax < 0.0f ) {
		return false;
	}

	// If tmin > tmax, ray does not intersect AABB.
	if ( tmin > tmax ) {
		return false;
	}

	t_near = tmin;
	t_far = tmax;
	return true;
}


////////////////////////////////////////////////////
// Fast, minimum storage ray/triangle intersection test.
// Implementation inspired by Tomas Moller: http://www.graphics.cornell.edu/pubs/1997/MT97.pdf
// Additional algorithm details: http://www.lighthouse3d.com/tutorials/maths/ray-triangle-intersection/
////////////////////////////////////////////////////
__device__
bool gpuTriIntersect( glm::vec3 ray_o, glm::vec3 ray_dir, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t, glm::vec3 &normal )
{
	glm::vec3 e1, e2, h, s, q;
	float a, f, u, v;

	e1 = v1 - v0;
	e2 = v2 - v0;

	h = glm::cross( ray_dir, e2 );
	a = glm::dot( e1, h );

	if ( a > -0.00001f && a < 0.00001f ) {
		return false;
	}

	f = 1.0f / a;
	s = ray_o - v0;
	u = f * glm::dot( s, h );

	if ( u < 0.0f || u > 1.0f ) {
		return false;
	}

	q = glm::cross( s, e1 );
	v = f * glm::dot( ray_dir, q );

	if ( v < 0.0f || u + v > 1.0f ) {
		return false;
	}

	// at this stage we can compute t to find out where the intersection point is on the line
	t = f * glm::dot( e2, q );

	if ( t > 0.00001f ) { // ray intersection
		normal = gpuComputeTriNormal( v0, v1, v2 );
		return true;
	}
	else { // this means that there is a line intersection but not a ray intersection
		return false;
	}
}


////////////////////////////////////////////////////
// computeTriNormal().
////////////////////////////////////////////////////
__device__
glm::vec3 gpuComputeTriNormal( const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3 )
{
	glm::vec3 u = p2 - p1;
	glm::vec3 v = p3 - p1;

	float nx = u.y * v.z - u.z * v.y;
	float ny = u.z * v.x - u.x * v.z;
	float nz = u.x * v.y - u.y * v.x;

	return glm::normalize( glm::vec3( nx, ny, nz ) );
}