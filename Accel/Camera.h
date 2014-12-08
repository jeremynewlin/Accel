#ifndef CAMERA_H
#define CAMERA_H

#include "../glm/glm.hpp"
#include "KDTreeStructs.h"


class Camera
{
public:
	Camera( float fovy, glm::vec2 reso, glm::vec3 eyep, glm::vec3 vdir, glm::vec3 uvec );
	~Camera( void );

	Ray computeRayThroughPixel( const int x, const int y ) const;

	glm::vec2 getResolution( void ) const;
	glm::vec3 getPosition( void ) const;
	glm::vec3 getM( void ) const;
	glm::vec3 getH( void ) const;
	glm::vec3 getV( void ) const;

private:
	float fovy, fovx;
	glm::vec2 reso;
	glm::vec3 eyep, vdir, uvec;
	glm::vec3 a, m, h, v;
};

#endif