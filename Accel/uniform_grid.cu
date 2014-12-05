#include "uniform_grid.h"

using namespace std;

glm::vec3 *cudapositions;
pair<int, int>* cudagrid;
int* cudaneighbors;
int* cudaids, *cudacellIds, *cudapIds;
int* cudanumNeighbors;

bool checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
	cout<<"Cuda error at "<<msg<<": "<<cudaGetErrorString(err)<<endl;
	cout<<endl;
	return false;
  }
  return true;
} 

__host__ __device__ 
float lengthSquared(glm::vec3 p){
    return p.x*p.x + p.y*p.y + p.z*p.z;
}

__device__ 
int hashParticle(glm::vec3 p, glm::vec3 gridSize, float h){
	int x = p.x/h;
	int y = p.y/h;
	int z = p.z/h;
	return x + y*gridSize.x + z*gridSize.x*gridSize.y;
}

__global__ 
void hashParticlesToGridKernel(int numParticles, int* cellIds, int* pIds, glm::vec3* positions, glm::vec3 gridSize, int* ids, float h){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index<numParticles){
		index = ids[index];
		cellIds[index] = hashParticle(positions[index], gridSize, h);
		pIds[index] = index;
	}
}

__global__ 
void resetGrid(int numGridCells, pair<int,int>* grid){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index<numGridCells){
		grid[index].first = -1;
		grid[index].second = -1;
	}
}


__global__ 
void setGridValuesKernel(int numParticles, int numGridCells, pair<int,int>* grid, int* cellIds){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index<numParticles){
		int cellId = cellIds[index];
		if (index==0){
			grid[cellId].first = index;
		}
		else{
			if (cellId == cellIds[index-1]){
				if (index > grid[cellId].second){
					grid[cellId].second = index;
				}
			}
			else{
				grid[cellId].first = index;
			}
		}
	}
}

__global__ 
void findNeighborsUsingGridKernel(int numParticles, float h, int maxNeighbors, glm::vec3* positions, 
								  int* neighbors, int* numNeighbors, pair<int,int>* grid, int* pIds, 
								  int numGridCells, int* ids, glm::vec3 gridSize, int* cellIds){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index<numParticles){
		index = ids[index];
		numNeighbors[index] = 0;

		int cellIdOriginal = hashParticle(positions[index], gridSize, h);
		if (cellIdOriginal<0 || cellIdOriginal>numGridCells-1) return;
		//search for neighbors in 3x3x3 neighbor grid cells
		//favors closer neighboring cells
		for (int cell=0; cell<27; cell++){
			int cellId1 = grid[cellIdOriginal].first;
			int cellId2 = grid[cellIdOriginal].second;
			
			if (numNeighbors[index]>=maxNeighbors){
				break;
			}

			int cellId = cellIdOriginal;

			if (cell==1){ //+x
				cellId = cellId+1;
			}
			else if (cell==2){ //-x
				cellId = cellId-1;
			}
			else if (cell==3){ //+y
				cellId = cellId+gridSize.x;
			}
			else if (cell==4){ //+z
				cellId = cellId+gridSize.x*gridSize.y;
			}
			else if (cell==5){ //-z
				cellId = cellId-gridSize.x*gridSize.y;
			}
			else if (cell==6){ //-y
				cellId = cellId-gridSize.x;
			}
			else if (cell==7){ //+x +y
				cellId = cellId+gridSize.x+1;
			}
			else if (cell==8){ //+x +z
				cellId = cellId+gridSize.x*gridSize.y+1;
			}
			else if (cell==9){ //+x -y
				cellId = cellId-gridSize.x+1;
			}
			else if (cell==10){ //+x -z
				cellId = cellId-gridSize.x*gridSize.y+1;
			}
			else if (cell==11){ //-x +y
				cellId = cellId+gridSize.x-1;
			}
			else if (cell==12){ //-x -y
				cellId = cellId-gridSize.x-1;
			}
			else if (cell==13){ //-x +z
				cellId = cellId+gridSize.x*gridSize.y-1;
			} 
			else if (cell==14){ //-x -z
				cellId = cellId-gridSize.x*gridSize.y-1;
			}
			else if (cell==15){ //+y +z
				cellId = cellId+gridSize.x+gridSize.x*gridSize.y;
			}
			else if (cell==16){ //+y -z
				cellId = cellId+gridSize.x-gridSize.x*gridSize.y;
			}
			else if (cell==17){ //-y +z
				cellId = cellId-gridSize.x+gridSize.x*gridSize.y;
			}
			else if (cell==18){ //-y -z
				cellId = cellId-gridSize.x-gridSize.x*gridSize.y;
			}
			else if (cell==19){ //+x +y +z
				cellId = cellId+gridSize.x*gridSize.y+gridSize.x+1;
			}
			else if (cell==20){ //+x -y +z
				cellId = cellId+gridSize.x*gridSize.y-gridSize.x+1;
			}
			else if (cell==21){ //+x +y -z
				cellId = cellId-gridSize.x*gridSize.y+gridSize.x+1;
			}
			else if (cell==22){ //+x -y -z
				cellId = cellId-gridSize.x*gridSize.y-gridSize.x+1;
			}
			else if (cell==23){ //-x +y +z
				cellId = cellId+gridSize.x*gridSize.y+gridSize.x-1;
			}
			else if (cell==24){ //-x -y +z
				cellId = cellId+gridSize.x*gridSize.y-gridSize.x-1;
			}
			else if (cell==25){ //-x +y -z
				cellId = cellId-gridSize.x*gridSize.y+gridSize.x-1;
			}
			else if (cell==26){ //-x -y -z
				cellId = cellId-gridSize.x*gridSize.y-gridSize.x-1;
			}

			//sanity check
			if (cellId<0 || cellId>numGridCells-1) continue;

			cellId1 = grid[cellId].first;
			if (cellId1==-1) continue;
			cellId2 = grid[cellId].second;
			if (cellId2==-1) cellId2=cellId1;
			
			if (cellId1>numParticles-1 || cellId1<0 || cellId2>numParticles-1 || cellId2<0) continue;

			int jid = ids[pIds[cellId1]];
			if (lengthSquared(positions[index]-positions[jid])<h*h && numNeighbors[index]<maxNeighbors){
				int nid = index*maxNeighbors+numNeighbors[index];
				neighbors[nid]=jid;
				numNeighbors[index]+=1;
			}

			for (int i=cellId1; i<cellId2+1; i+=1){
				int jid = ids[pIds[i]];
				if (lengthSquared(positions[index]-positions[jid])<h*h && numNeighbors[index]<maxNeighbors){
					int nid = index*maxNeighbors+numNeighbors[index];
					neighbors[nid]=jid;
					numNeighbors[index]+=1;
				}
			}
		}
	}
}

__global__
void findNeighborsKernel(int numParticles, glm::vec3* positions, int* neighbors, int* numNeighbors, float h, int maxNeighbors, int* ids){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index<numParticles){
		index = ids[index];
		numNeighbors[index] = 0;
		for (int j=0; j<numParticles; j++){
			if (lengthSquared(positions[index]-positions[j])<h*h && numNeighbors[index]<maxNeighbors){
				neighbors[index*maxNeighbors+numNeighbors[index]]=ids[j];
				numNeighbors[index]+=1;
			}
		}
	}
}

__global__
void resetNumNeighbors(int numParticles, int* numNeighbors){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index<numParticles){
		numNeighbors[index] = 0;
	}
}


hash_grid::hash_grid(int numParticles, glm::vec3* points, glm::vec3 gridSize){

	m_numParticles = numParticles;
	m_points = new glm::vec3[m_numParticles];
	m_ids = new int[m_numParticles];
	for (int i=0; i<m_numParticles; i+=1){
		m_points[i] = points[i];
		m_ids[i] = i;
	}

	m_gridSize = gridSize;

	m_grid = new pair<int,int>[int(m_gridSize.x*m_gridSize.y*m_gridSize.z)];
	m_cellIds = new int[m_numParticles];
	m_pIds = new int[m_numParticles];

	c_positions = NULL;
	cudaMalloc((void**)&c_positions, m_numParticles*sizeof(glm::vec3));
	cudaMemcpy( c_positions, m_points, m_numParticles*sizeof(glm::vec3), cudaMemcpyHostToDevice);
	
	c_ids = NULL;
	cudaMalloc((void**)&c_ids, m_numParticles*sizeof(int));
	cudaMemcpy( c_ids, m_ids, m_numParticles*sizeof(int), cudaMemcpyHostToDevice);

	c_ids_test = NULL;
	cudaMalloc((void**)&c_ids_test, m_numParticles*sizeof(int));
	cudaMemcpy( c_ids_test, m_ids, m_numParticles*sizeof(int), cudaMemcpyHostToDevice);

	c_cellIds = NULL;
	cudaMalloc((void**)&c_cellIds, m_numParticles*sizeof(int));

	c_pIds = NULL;
	cudaMalloc((void**)&c_pIds, m_numParticles*sizeof(int));

	c_grid = NULL;
	cudaMalloc((void**)&c_grid, int(m_gridSize.x*m_gridSize.y*m_gridSize.z)*sizeof(pair<int,int>));

	neighborsAlloc = false;
	m_maxNeighbors = -1;
}

void hash_grid::findNeighbors(int maxNeighbors, float h, bool useGrid, bool useGPU){
	if (maxNeighbors < 0){
		return;
	}

	if (m_maxNeighbors != maxNeighbors && m_maxNeighbors != -1){
		cudaFree(c_numNeighbors);
		cudaFree(c_neighbors);
		delete [] m_gridNeighbors;
		delete [] m_bruteNeighbors;
		delete [] m_gridNumNeighbors;
		delete [] m_bruteNumNeighbors;
	}

	m_maxNeighbors = maxNeighbors;
	m_h = h;

	m_gridNeighbors = new int[m_numParticles*m_maxNeighbors];
	m_bruteNeighbors = new int[m_numParticles*m_maxNeighbors];
	m_gridNumNeighbors = new int[m_numParticles];
	m_bruteNumNeighbors = new int[m_numParticles];

	neighborsAlloc = true;

	c_neighbors = NULL;
	cudaMalloc((void**)&c_neighbors, m_numParticles*m_maxNeighbors*sizeof(int*));

	c_numNeighbors = NULL;
	cudaMalloc((void**)&c_numNeighbors, m_numParticles*sizeof(int));

	if (useGPU){
		findNeighborsGPU(useGrid);
	}
	else{
		findNeighborsCPU(useGrid);
	}
}

void hash_grid::findNeighborsGPU(bool useGrid){
	
	dim3 threadsPerBlock(64);
	dim3 fullBlocksPerGrid(m_numParticles/8+1);

	dim3 threadsPerBlockGrid(64);
	dim3 fullBlocksPerGridGrid(int(m_gridSize.x*m_gridSize.y*m_gridSize.z)/8+1);

	hashParticlesToGridKernel<<<fullBlocksPerGrid, threadsPerBlock>>>
		(m_numParticles, c_cellIds, c_pIds, c_positions, m_gridSize, c_ids, m_h);
	cudaThreadSynchronize();
	
	checkCUDAError("hashing particles");

	thrust::device_ptr<int> thrustCellIds = thrust::device_pointer_cast(c_cellIds);
	thrust::device_ptr<int> thrustPIds = thrust::device_pointer_cast(c_pIds);
	thrust::device_ptr<glm::vec3> thrustPos = thrust::device_pointer_cast(c_positions);

	thrust::sort_by_key(thrustCellIds, thrustCellIds+m_numParticles, thrustPIds);
	thrust::sort_by_key(thrustCellIds, thrustCellIds+m_numParticles, thrustPos);

	resetGrid<<<fullBlocksPerGridGrid, threadsPerBlockGrid>>>
		(int(m_gridSize.x*m_gridSize.y*m_gridSize.z), c_grid);
	
	checkCUDAError("reset grid");
	
	setGridValuesKernel<<<fullBlocksPerGrid, threadsPerBlock>>>
		(m_numParticles, int(m_gridSize.x*m_gridSize.y*m_gridSize.z), c_grid, c_cellIds);
	
	checkCUDAError("set values in grid neighbor");
	
	if (useGrid){
		resetNumNeighbors<<<fullBlocksPerGrid, threadsPerBlock>>>
			(m_numParticles, c_numNeighbors);
		findNeighborsUsingGridKernel<<<fullBlocksPerGrid, threadsPerBlock>>>
			(m_numParticles, m_h, m_maxNeighbors, c_positions, c_neighbors, c_numNeighbors, c_grid, 
			c_pIds, int(m_gridSize.x*m_gridSize.y*m_gridSize.z), c_ids, m_gridSize, c_cellIds);
	
		cudaMemcpy( m_gridNumNeighbors, c_numNeighbors, m_numParticles*sizeof(int), cudaMemcpyDeviceToHost);
		checkCUDAError(" copying num neighbor dbhtrwhnwe ");

		cudaMemcpy( m_gridNeighbors, c_neighbors, m_numParticles*m_maxNeighbors*sizeof(int), cudaMemcpyDeviceToHost);
		checkCUDAError(" copying neighbor dbhtrwhnwe ");

		int avg = 0;
		for (int i=0; i<m_numParticles; i++){
			avg+=m_gridNumNeighbors[i];
		}

		cout<<"average number of neighbors with grid: "<<float(avg)/float(m_numParticles)<<endl;
	}
	//////////////////
	else{
		resetNumNeighbors<<<fullBlocksPerGrid, threadsPerBlock>>>
			(m_numParticles, c_numNeighbors);
		findNeighborsKernel<<<fullBlocksPerGrid, threadsPerBlock>>>
			(m_numParticles, c_positions, c_neighbors, c_numNeighbors, m_h, m_maxNeighbors, c_ids);
	
		cudaMemcpy( m_bruteNumNeighbors, c_numNeighbors, m_numParticles*sizeof(int), cudaMemcpyDeviceToHost);
		checkCUDAError(" copying num neighbor dbhtrwhnwe ");

		cudaMemcpy( m_bruteNeighbors, c_neighbors, m_numParticles*m_maxNeighbors*sizeof(int), cudaMemcpyDeviceToHost);
		checkCUDAError(" copying neighbor dbhtrwhnwe ");

		int avg = 0;
		for (int i=0; i<m_numParticles; i++){
			avg+=m_bruteNumNeighbors[i];
		}

		cout<<"average number of neighbors with brute: "<<float(avg)/float(m_numParticles)<<endl;
	}
	checkCUDAError(" finding neighbors using grid ");
}

void hash_grid::findNeighborsCPU(bool useGrid){
	for (int index = 0; index<m_numParticles; index+=1){
		m_gridNumNeighbors[index] = 0;
		m_bruteNumNeighbors[index] = 0;
	}

	for (int i = 0; i<m_numParticles; i+=1){
		int index = m_ids[i];
		m_cellIds[index] = hashParticle(index);
		m_pIds[index] = index;
	}

	thrust::device_ptr<int> thrustCellIds = thrust::device_pointer_cast(m_cellIds);
	thrust::device_ptr<int> thrustPIds = thrust::device_pointer_cast(m_pIds);
	thrust::sort_by_key(m_cellIds, m_cellIds+m_numParticles, m_pIds);
	
	int numGridCells = int(m_gridSize.x * m_gridSize.y * m_gridSize.z);
	for (int index = 0; index<numGridCells; index+=1){
		m_grid[index].first = -1;
		m_grid[index].second = -1;
	}

	for (int index = 0; index<m_numParticles; index+=1){
		if (index<m_numParticles){
			int cellId = m_cellIds[index];
			if (index==0){
				m_grid[cellId].first = index;
			}
			else{
				if (cellId == m_cellIds[index-1]){
					if (index > m_grid[cellId].second){
						m_grid[cellId].second = index;
					}
				}
				else{
					m_grid[cellId].first = index;
				}
			}
		}
	}

	if (useGrid){
		findNeighborsCPUGridHelper();
	}
	else{
		findNeighborsCPUBruteHelper();
	}
}

void hash_grid::findNeighborsCPUGridHelper(){
	
	int numGridCells = int(m_gridSize.x * m_gridSize.y * m_gridSize.z);

	for (int i=0; i<m_numParticles; i+=1){
		int index = m_ids[i];
		m_gridNumNeighbors[index] = 0;

		int cellIdOriginal = hashParticle(index);
		if (cellIdOriginal<0 || cellIdOriginal>numGridCells-1) return;
		//search for m_gridNeighbors in 3x3x3 neighbor grid cells
		//favors closer neighboring cells
		for (int cell=0; cell<27; cell++){
			int cellId1 = m_grid[cellIdOriginal].first;
			int cellId2 = m_grid[cellIdOriginal].second;
			
			if (m_gridNumNeighbors[index]>=m_maxNeighbors){
				break;
			}

			int cellId = cellIdOriginal;

			if (cell==1){ //+x
				cellId = cellId+1;
			}
			else if (cell==2){ //-x
				cellId = cellId-1;
			}
			else if (cell==3){ //+y
				cellId = cellId+m_gridSize.x;
			}
			else if (cell==4){ //+z
				cellId = cellId+m_gridSize.x*m_gridSize.y;
			}
			else if (cell==5){ //-z
				cellId = cellId-m_gridSize.x*m_gridSize.y;
			}
			else if (cell==6){ //-y
				cellId = cellId-m_gridSize.x;
			}
			else if (cell==7){ //+x +y
				cellId = cellId+m_gridSize.x+1;
			}
			else if (cell==8){ //+x +z
				cellId = cellId+m_gridSize.x*m_gridSize.y+1;
			}
			else if (cell==9){ //+x -y
				cellId = cellId-m_gridSize.x+1;
			}
			else if (cell==10){ //+x -z
				cellId = cellId-m_gridSize.x*m_gridSize.y+1;
			}
			else if (cell==11){ //-x +y
				cellId = cellId+m_gridSize.x-1;
			}
			else if (cell==12){ //-x -y
				cellId = cellId-m_gridSize.x-1;
			}
			else if (cell==13){ //-x +z
				cellId = cellId+m_gridSize.x*m_gridSize.y-1;
			} 
			else if (cell==14){ //-x -z
				cellId = cellId-m_gridSize.x*m_gridSize.y-1;
			}
			else if (cell==15){ //+y +z
				cellId = cellId+m_gridSize.x+m_gridSize.x*m_gridSize.y;
			}
			else if (cell==16){ //+y -z
				cellId = cellId+m_gridSize.x-m_gridSize.x*m_gridSize.y;
			}
			else if (cell==17){ //-y +z
				cellId = cellId-m_gridSize.x+m_gridSize.x*m_gridSize.y;
			}
			else if (cell==18){ //-y -z
				cellId = cellId-m_gridSize.x-m_gridSize.x*m_gridSize.y;
			}
			else if (cell==19){ //+x +y +z
				cellId = cellId+m_gridSize.x*m_gridSize.y+m_gridSize.x+1;
			}
			else if (cell==20){ //+x -y +z
				cellId = cellId+m_gridSize.x*m_gridSize.y-m_gridSize.x+1;
			}
			else if (cell==21){ //+x +y -z
				cellId = cellId-m_gridSize.x*m_gridSize.y+m_gridSize.x+1;
			}
			else if (cell==22){ //+x -y -z
				cellId = cellId-m_gridSize.x*m_gridSize.y-m_gridSize.x+1;
			}
			else if (cell==23){ //-x +y +z
				cellId = cellId+m_gridSize.x*m_gridSize.y+m_gridSize.x-1;
			}
			else if (cell==24){ //-x -y +z
				cellId = cellId+m_gridSize.x*m_gridSize.y-m_gridSize.x-1;
			}
			else if (cell==25){ //-x +y -z
				cellId = cellId-m_gridSize.x*m_gridSize.y+m_gridSize.x-1;
			}
			else if (cell==26){ //-x -y -z
				cellId = cellId-m_gridSize.x*m_gridSize.y-m_gridSize.x-1;
			}

			//sanity check
			if (cellId<0 || cellId>numGridCells-1) continue;

			cellId1 = m_grid[cellId].first;
			if (cellId1==-1) continue;
			cellId2 = m_grid[cellId].second;
			if (cellId2==-1) cellId2=cellId1;
			
			if (cellId1>m_numParticles-1 || cellId1<0 || cellId2>m_numParticles-1 || cellId2<0) continue;

			int jid = m_ids[m_pIds[cellId1]];
			if (lengthSquared(m_points[index]-m_points[jid])<m_h*m_h && m_gridNumNeighbors[index]<m_maxNeighbors){
				int nid = index*m_maxNeighbors+m_gridNumNeighbors[index];
				m_gridNeighbors[nid]=jid;
				m_gridNumNeighbors[index]+=1;
			}

			for (int i=cellId1; i<cellId2+1; i+=1){
				int jid = m_ids[m_pIds[i]];
				if (lengthSquared(m_points[index]-m_points[jid])<m_h*m_h && m_gridNumNeighbors[index]<m_maxNeighbors){
					int nid = index*m_maxNeighbors+m_gridNumNeighbors[index];
					m_gridNeighbors[nid]=jid;
					m_gridNumNeighbors[index]+=1;
				}
			}
		}
	}
}

void hash_grid::findNeighborsCPUBruteHelper(){
	for (int i=0; i<m_numParticles; i+=1){
		glm::vec3 point = m_points[i];
		m_bruteNumNeighbors[i] = 0;
		for (int j=0; j<m_numParticles && m_bruteNumNeighbors[i]<m_maxNeighbors; j+=1){
			if (lengthSquared(point - m_points[j]) < m_h*m_h){
				m_bruteNeighbors[i*m_maxNeighbors+m_bruteNumNeighbors[i]]=m_ids[j];
				m_bruteNumNeighbors[i] += 1;
			}
		}
	}
}

int hash_grid::hashParticle(int id) const{
	glm::vec3 p = m_points[id];
	int x = p.x/m_h;
	int y = p.y/m_h;
	int z = p.z/m_h;
	return x + y*m_gridSize.x + z*m_gridSize.x*m_gridSize.y;
}

hash_grid::~hash_grid(){

	delete [] m_points;
	delete [] m_ids;
	delete [] m_grid;
	delete [] m_cellIds;
	delete [] m_pIds;

	if (neighborsAlloc){
		cudaFree(c_numNeighbors);
		cudaFree(c_neighbors);
		delete [] m_gridNeighbors;
		delete [] m_bruteNeighbors;
		delete [] m_gridNumNeighbors;
		delete [] m_bruteNumNeighbors;
	}

	cudaFree(c_positions);
	cudaFree(c_cellIds);
	cudaFree(c_pIds);
	cudaFree(c_grid);
	cudaFree(c_ids);
	cudaFree(c_ids_test);
}