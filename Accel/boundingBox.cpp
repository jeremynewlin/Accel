#include "boundingBox.h"

boundingBox::boundingBox(){
}

// Constructor.
boundingBox::boundingBox( std::vector<Triangle*> t )
{
	glm::vec3 tmp_max = glm::vec3( -10000.0f, -10000.0f, -10000.0f );
	glm::vec3 tmp_min = glm::vec3( 10000.0f, 10000.0f, 10000.0f );

	for ( std::vector<Triangle*>::iterator it = t.begin(); it != t.end(); ++it ) {
		Triangle *tri = *it;
		glm::vec3 tri_max = tri->getMax();
		glm::vec3 tri_min = tri->getMin();

		// Update tmp_max.
		if ( tri_max.x > tmp_max.x ) {
			tmp_max.x = tri_max.x;
		}
		if ( tri_max.y > tmp_max.y ) {
			tmp_max.y = tri_max.y;
		}
		if ( tri_max.z > tmp_max.z ) {
			tmp_max.z = tri_max.z;
		}

		// Update tmp_min.
		if ( tri_min.x < tmp_min.x ) {
			tmp_min.x = tri_min.x;
		}
		if ( tri_min.y < tmp_min.y ) {
			tmp_min.y = tri_min.y;
		}
		if ( tri_min.z < tmp_min.z ) {
			tmp_min.z = tri_min.z;
		}
	}

	max = tmp_max;
	min = tmp_min;
}