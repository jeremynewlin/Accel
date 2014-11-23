#ifndef UNIFORM_GRID_H
#define UNIFORM_GRID_H

#include <vector>
#include "../glm/glm.hpp"
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

void initCuda(int numParticles, int* ids, glm::vec3* positions, int maxNeighbors, glm::vec3 gridSize);
void findNeighbors(int numParticles, int maxNeighbors, glm::vec3 gridSize, float h);
void freeCudaGrid();

void test_uniform_grid();

class hash_grid{
public:
	int m_numParticles;
	glm::vec3* m_points;
	glm::vec3 m_gridSize;
	int* m_ids;
	int m_maxNeighbors;

public:
	hash_grid(int numParticles, glm::vec3* points, glm::vec3 gridSize);

	void findNeighbors(int maxNeighbors, float h);

	~hash_grid();

private:

	bool neighborsAlloc;

	glm::vec3 *c_positions;
	std::pair<int, int>* c_grid;
	int* c_neighbors;
	int* c_ids, *c_cellIds, *c_pIds;
	int* c_numNeighbors;

};

#endif