#ifndef KDTREE_H
#define KDTREE_H

#include "boundingBox.h"
#include "mesh.h"

class node{

	node* left;
	node* right;

};

class kdtree{

public:
	kdtree(mesh* m);
	~kdtree();
	mesh* m_mesh;

	void construct();

private:
	glm::vec3* cudaTris;
	glm::vec3* cudaVerts;

	void perTriBoundingBox();

//for debugging only
public:
	boundingBox* boundingBoxes;
	boundingBox* cudaBoundingBoxes;

};

#endif