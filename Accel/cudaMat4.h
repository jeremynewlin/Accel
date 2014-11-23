// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef CUDAMAT4_H
#define CUDAMAT4_H

#include "../glm/glm.hpp"
#include <cuda_runtime.h>

struct cudaMat3{
	glm::vec3 x;
	glm::vec3 y;
	glm::vec3 z;
};

struct cudaMat4{
	glm::vec4 x;
	glm::vec4 y;
	glm::vec4 z;
	glm::vec4 w;
};

#endif