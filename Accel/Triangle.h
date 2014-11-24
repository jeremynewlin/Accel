#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "../glm/glm.hpp"

class Triangle
{
public:
	Triangle( void );
	Triangle( glm::vec3 v1, glm::vec3 v2, glm::vec3 v3 )
	{
		this->v1 = v1;
		this->v2 = v2;
		this->v3 = v3;
		center = ( ( v1 + v2 + v3 ) / 3.0f );
	}

	// Find max x, y, z in triangle.
	glm::vec3 getMax( void ) const
	{
		glm::vec3 max;
		max.x = ( v1.x > v2.x && v1.x > v3.x ) ? v1.x : ( v2.x > v3.x ? v2.x : v3.x );
		max.y = ( v1.y > v2.y && v1.y > v3.y ) ? v1.y : ( v2.y > v3.y ? v2.y : v3.y );
		max.z = ( v1.z > v2.z && v1.z > v3.z ) ? v1.z : ( v2.z > v3.z ? v2.z : v3.z );
		return max;
	}

	// Find min x, y, z in triangle.
	glm::vec3 getMin( void ) const
	{
		glm::vec3 min;
		min.x = ( v1.x < v2.x && v1.x < v3.x ) ? v1.x : ( v2.x < v3.x ? v2.x : v3.x );
		min.y = ( v1.y < v2.y && v1.y < v3.y ) ? v1.y : ( v2.y < v3.y ? v2.y : v3.y );
		min.z = ( v1.z < v2.z && v1.z < v3.z ) ? v1.z : ( v2.z < v3.z ? v2.z : v3.z );
		return min;
	}

	glm::vec3 v1;
	glm::vec3 v2;
	glm::vec3 v3;

	glm::vec3 center;
};

#endif