#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "boundingBox.h"
#include "../glm/glm.hpp"


float const EPSILON = pow( 1.0f, -10 );

enum AABBDir {
	RIGHT = 0,
	LEFT = 1,
	MIDDLE = 2
};


class Intersections
{
public:
	Intersections( void );
	~Intersections( void );

	static bool AABBIntersect( boundingBox bbox, glm::vec3 ray_o, glm::vec3 ray_dir, glm::vec3 &hit_point )
};

#endif