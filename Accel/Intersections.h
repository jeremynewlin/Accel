#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "boundingBox.h"
#include "../glm/glm.hpp"
#include <cuda_runtime.h>


class Intersections
{
public:
	Intersections( void );
	~Intersections( void );

	__host__ __device__ static bool aabbIntersect( boundingBox bbox, glm::vec3 ray_o, glm::vec3 ray_dir, float &t_near, float &t_far );
	__host__ __device__ static bool triIntersect( glm::vec3 ray_o, glm::vec3 ray_dir, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t, glm::vec3 &normal );

	__host__ __device__ static glm::vec3 computeTriNormal( const glm::vec3&, const glm::vec3&, const glm::vec3& );
};

#endif