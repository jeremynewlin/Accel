#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "boundingBox.h"
#include "../glm/glm.hpp"


const float INTERSECTION_EPSILON = pow( 1.0f, -5 );


class Intersections
{
public:
	Intersections( void );
	~Intersections( void );

	static bool aabbIntersect( boundingBox bbox, glm::vec3 ray_o, glm::vec3 ray_dir, float &t_near, float &t_far );
	static bool triIntersect( glm::vec3 ray_o, glm::vec3 ray_dir, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t, glm::vec3 &normal );

	static glm::vec3 computeTriNormal( const glm::vec3&, const glm::vec3&, const glm::vec3& );
};

#endif