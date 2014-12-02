#include "Intersections.h"


////////////////////////////////////////////////////
// Constructor/destructor.
////////////////////////////////////////////////////

Intersections::Intersections()
{
}

Intersections::~Intersections()
{
}


////////////////////////////////////////////////////
// Fast ray/AABB intersection test.
// Implementation inspired by zacharmarz.
// https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
////////////////////////////////////////////////////
bool Intersections::AABBIntersect( boundingBox bbox, glm::vec3 ray_o, glm::vec3 ray_dir, glm::vec3 &hit_point )
{
	glm::vec3 dirfrac( 1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z );

	float t1 = ( bbox.min.x - ray_o.x ) * dirfrac.x;
	float t2 = ( bbox.max.x - ray_o.x ) * dirfrac.x;
	float t3 = ( bbox.min.y - ray_o.y ) * dirfrac.y;
	float t4 = ( bbox.max.y - ray_o.y ) * dirfrac.y;
	float t5 = ( bbox.min.z - ray_o.z ) * dirfrac.z;
	float t6 = ( bbox.max.z - ray_o.z ) * dirfrac.z;

	float tmin = std::max( std::max( std::min( t1, t2 ), std::min( t3, t4 ) ), std::min( t5, t6 ) );
	float tmax = std::min( std::min( std::max( t1, t2 ), std::max( t3, t4 ) ), std::max( t5, t6 ) );

	// If tmax < 0, ray intersects AABB, but entire AABB is behing ray, so reject.
	if ( tmax < 0.0f ) {
		return false;
	}

	// If tmin > tmax, ray does not intersect AABB.
	if ( tmin > tmax ) {
		return false;
	}

	hit_point =  ray_o + ( tmin * ray_dir );
	return true;
}


////////////////////////////////////////////////////
// Fast, minimum storage ray/triangle intersection test.
// Implementation inspired by Tomas Moller.
// http://www.graphics.cornell.edu/pubs/1997/MT97.pdf
////////////////////////////////////////////////////
bool Intersections::TriIntersect( glm::vec3 ray_o, glm::vec3 ray_dir, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 &hit_point )
{
    glm::vec3 edge1, edge2, tvec, pvec, qvec;
    float det, inv_det;
    float t, u, v;

    // Find vectors for two edges sharing v0.
    edge1 = v1 - v0;
    edge2 = v2 - v0;

    // Compute determinant.
    pvec = glm::cross( ray_dir, edge2 );
    det = glm::dot( edge1, pvec );

    // If determinant is 0, then ray lies in plane of triangle.
    if ( det < TEST_EPSILON ) {
        return false;
    }

    // Compute u parameter and test bounds.
    tvec = ray_o - v0;
    u = glm::dot( tvec, pvec );
    if ( u < 0.0f || u > det ) {
        return false;
    }

    // Compute v parameter and test bounds.
    qvec = glm::cross( tvec, edge1 );
    v = glm::dot( ray_dir, qvec );
    if ( v < 0.0f || ( u + v ) > det ) {
        return false;
    }

    // Compute t. Scale t. Ray intersects triangle.
    t = glm::dot( edge2, qvec );
    inv_det = 1.0f / det;
    t *= inv_det;

    // Compute hit_point.
    hit_point = ray_o + ( t * ray_dir );

    return true;
}