#include "kdtree.h"


kdtree::kdtree(mesh* m){
	this->m_mesh = m;
	
	this->cudaTris = NULL;
	cudaMalloc((void**)&cudaTris, m->numTris*sizeof(glm::vec3));

	this->cudaVerts = NULL;
	cudaMalloc((void**)&cudaVerts, m->numVerts*sizeof(glm::vec3));


	//debug
	this->boundingBoxes = new boundingBox[m->numTris];
	cudaBoundingBoxes = NULL;
	cudaMalloc((void**)&cudaBoundingBoxes, m->numTris*sizeof(boundingBox));
}

kdtree::~kdtree(){
	
	cudaFree(this->cudaTris);
	cudaFree(this->cudaVerts);

	delete [] this->boundingBoxes;
	cudaFree(this->cudaBoundingBoxes);
}