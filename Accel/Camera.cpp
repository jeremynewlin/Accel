#include "utils.h"
#include "Camera.h"


////////////////////////////////////////////////////
// Constructor/destructor.
////////////////////////////////////////////////////

Camera::Camera( float fovy, glm::vec2 reso, glm::vec3 eyep, glm::vec3 vdir, glm::vec3 uvec )
{
	this->fovy = fovy;
	this->reso = reso;
	this->eyep = eyep;
	this->vdir = vdir;
	this->uvec = uvec;

	// Field-of-view in the x direction.
	fovx = ( ( fovy * reso.x ) / reso.y );

	// Cross product of C and U.
	a = glm::cross( vdir, uvec );
	
	// Midpoint of frame buffer.
	m = eyep + vdir;

	// Horizontal NDC value. Parallel to A.
	h = ( a * glm::length( vdir ) * tan( utilityCore::degreesToRadians( fovx ) ) ) / glm::length( a );
	
	// Vertical NDC value. Parallel to B.
	v = glm::vec3( 0.0f, reso.y * glm::length( h ) / reso.x, 0.0f );
}

Camera::~Camera()
{
}


////////////////////////////////////////////////////
// Compute ray from camera through passed-in pixel coordinate.
////////////////////////////////////////////////////
Ray Camera::computeRayThroughPixel( const int x, const int y ) const
{
	float sx = x / reso.x;
	float sy = 1 - ( y / reso.y );
	glm::vec3 point_in_3d_space = m + ( 2.0f * sx - 1.0f ) * h + ( 2.0f * sy - 1.0f ) * v;

	Ray r;
	r.origin = eyep;
	r.dir = glm::normalize( point_in_3d_space - eyep );

	return r;
}


////////////////////////////////////////////////////
// Getters.
////////////////////////////////////////////////////

glm::vec2 Camera::getResolution() const
{
	return reso;
}