#ifndef BB_H
#define BB_H

#include "../glm/glm.hpp"
#include "Triangle.h"
#include <vector>

enum Axis {
	XAXIS = 0,
	YAXIS = 1,
	ZAXIS = 2
};

class boundingBox
{
public:
	boundingBox();
	boundingBox( std::vector<Triangle*> t );

	glm::vec3 min, max;

	Axis getLongestSide();
};

#endif