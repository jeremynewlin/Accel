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
// Fast ray-box intersection test.
// Implementation inspired by Andrew Woo from "Graphics Gems", Academic Press, 1990.
////////////////////////////////////////////////////
bool Intersections::AABBIntersect( boundingBox bbox, glm::vec3 ray_o, glm::vec3 ray_dir, glm::vec3 &hit_point )
{
	glm::vec3 minB = bbox.min;
	glm::vec3 maxB = bbox.max;
	bool inside = true;
	AABBDir quadrant[3];
	int i;
	int whichPlane;
	glm::vec3 maxT( 0.0f, 0.0f, 0.0f );
	glm::vec3 candidatePlane( 0.0f, 0.0f, 0.0f );

	// Find candidate planes.
	// This loop can be avoided if rays cast all from the eye (assume perpsective view).
	for ( i = 0; i < 3; ++i ) {
		if ( ray_o[i] < minB[i] ) {
			quadrant[i] = LEFT;
			candidatePlane[i] = minB[i];
			inside = false;
		}
		else if ( ray_o[i] > maxB[i] ) {
			quadrant[i] = RIGHT;
			candidatePlane[i] = maxB[i];
			inside = false;
		}
		else {
			quadrant[i] = MIDDLE;
		}
	}

	// Ray origin inside bounding box.
	if ( inside ) {
		hit_point = ray_o;
		return true;
	}

	// Compute T distances to candidate planes.
	for ( i = 0; i < 3; ++i ) {
		if ( quadrant[i] != MIDDLE && ray_dir[i] > EPSILON && ray_dir[i] < -EPSILON ) {
			maxT[i] = ( candidatePlane[i] - ray_o[i]) / ray_dir[i];
		}
		else {
			maxT[i] = -1.0f;
		}
	}

	// Get largest of the maxT's for final choice of intersection.
	whichPlane = 0;
	for ( i = 1; i < 3; i++ ) {
		if ( maxT[whichPlane] < maxT[i] ) {
			whichPlane = i;
		}
	}

	// Check final candidate actually inside box.
	if ( maxT[whichPlane] < 0.0f ) {
		return false;
	}
	for ( i = 0; i < 3; i++ ) {
		if ( whichPlane != i ) {
			hit_point[i] = ray_o[i] + ( maxT[whichPlane] * ray_dir[i] );
			if ( hit_point[i] < minB[i] || hit_point[i] > maxB[i] ) {
				return false;
			}
		}
		else {
			hit_point[i] = candidatePlane[i];
		}
	}

	// Ray hits box.
	return true;
}