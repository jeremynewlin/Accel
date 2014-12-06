#include "Intersections.h"
#include <algorithm>


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
bool Intersections::aabbIntersect( boundingBox bbox, glm::vec3 ray_o, glm::vec3 ray_dir, float &t_near, float &t_far )
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
bool Intersections::triIntersect( glm::vec3 ray_o, glm::vec3 ray_dir, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t, glm::vec3 &normal )
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
		normal = Intersections::computeTriNormal( v0, v1, v2 );
		return true;
	}
	else { // this means that there is a line intersection but not a ray intersection
		return false;
	}
}


////////////////////////////////////////////////////
// computeTriNormal().
////////////////////////////////////////////////////
glm::vec3 Intersections::computeTriNormal( const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3 )
{
	glm::vec3 u = p2 - p1;
	glm::vec3 v = p3 - p1;

	float nx = u.y * v.z - u.z * v.y;
	float ny = u.z * v.x - u.x * v.z;
	float nz = u.x * v.y - u.y * v.x;

	return glm::normalize( glm::vec3( nx, ny, nz ) );
}