#include "kdtree.h"


kdtree::kdtree(mesh* m){
	this->m_mesh = m;
	
	this->cudaTris = NULL;
	cudaMalloc((void**)&cudaTris, m->numTris*sizeof(glm::vec3));
	cudaMemcpy( cudaTris, m->tris, m->numTris*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	this->cudaVerts = NULL;
	cudaMalloc((void**)&cudaVerts, m->numVerts*sizeof(glm::vec3));
	cudaMemcpy( cudaVerts, m->verts, m->numVerts*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	//debug
	this->boundingBoxes = new boundingBox[m->numTris];
	this->cudaBoundingBoxes = NULL;
	cudaMalloc((void**)&cudaBoundingBoxes, m->numTris*sizeof(boundingBox));
	cudaMemcpy( cudaBoundingBoxes, this->boundingBoxes, m->numTris*sizeof(boundingBox), cudaMemcpyHostToDevice);
}

kdtree::~kdtree(){
	
	cudaFree(this->cudaTris);
	cudaFree(this->cudaVerts);

	delete [] this->boundingBoxes;
	cudaFree(this->cudaBoundingBoxes);
}

void kdtree::construct(){
	perTriBoundingBox();
}

__global__ 
void perTriBoundingBoxKernel(int numTris, glm::vec3* tris, glm::vec3* verts, boundingBox* bbs){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index<numTris){
		glm::vec3 tri = tris[index];
		int ix = int(tri.x);
		int iy = int(tri.y);
		int iz = int(tri.z);

		glm::vec3 p1 = verts[ix];
		glm::vec3 p2 = verts[iy];
		glm::vec3 p3 = verts[iz];

		float maxX = glm::max(p1.x, p2.x);
		maxX = glm::max(maxX, p3.x);

		float maxY = glm::max(p1.y, p2.y);
		maxY = glm::max(maxY, p3.y);

		float maxZ = glm::max(p1.z, p2.z);
		maxZ = glm::max(maxZ, p3.z);

		float minX = glm::min(p1.x, p2.x);
		minX = glm::min(minX, p3.x);

		float minY = glm::min(p1.y, p2.y);
		minY = glm::min(minY, p3.y);

		float minZ = glm::min(p1.z, p2.z);
		minZ = glm::min(minZ, p3.z);

		bbs[index].max = glm::vec3(maxX,maxY,maxZ);
		bbs[index].min = glm::vec3(minX,minY,minZ);
	}
}

void kdtree::perTriBoundingBox(){
	dim3 threadsPerBlock(64);
	dim3 fullBlocksPerGrid(m_mesh->numTris/8+1);

	perTriBoundingBoxKernel<<<fullBlocksPerGrid, threadsPerBlock>>>
		(this->m_mesh->numTris, this->cudaTris, this->cudaVerts, this->cudaBoundingBoxes);

	cudaMemcpy( this->boundingBoxes, cudaBoundingBoxes, this->m_mesh->numTris*sizeof(boundingBox), cudaMemcpyDeviceToHost);
}