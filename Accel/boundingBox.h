#ifndef BB_H
#define BB_H

#include "../glm/glm.hpp"
#include "Triangle.h"
#include <vector>

class boundingBox
{
public:
	boundingBox();
	boundingBox( std::vector<Triangle*> t );

	glm::vec3 min, max;
};

#endif