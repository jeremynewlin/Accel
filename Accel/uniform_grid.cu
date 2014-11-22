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
    //fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
	cout<<"Cuda error at "<<msg<<": "<<cudaGetErrorString(err)<<endl;
	cout<<endl;
	return false;
  }
  return true;
} 

__device__ 
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
void hello(int *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}

void initCuda(int numParticles, int* ids, glm::vec3* positions, int maxNeighbors, glm::vec3 gridSize){
	cudapositions = NULL;
	cudaMalloc((void**)&cudapositions, numParticles*sizeof(glm::vec3));
	cudaMemcpy( cudapositions, positions, numParticles*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaneighbors = NULL;
	cudaMalloc((void**)&cudaneighbors, numParticles*maxNeighbors*sizeof(int*));

	cudanumNeighbors = NULL;
	cudaMalloc((void**)&cudanumNeighbors, numParticles*sizeof(int));
	cudaMemcpy( cudanumNeighbors, ids, numParticles*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaids = NULL;
	cudaMalloc((void**)&cudaids, numParticles*sizeof(int));
	cudaMemcpy( cudaids, ids, numParticles*sizeof(int), cudaMemcpyHostToDevice);

	cudacellIds = NULL;
	cudaMalloc((void**)&cudacellIds, numParticles*sizeof(int));

	cudapIds = NULL;
	cudaMalloc((void**)&cudapIds, numParticles*sizeof(int));

	cudagrid = NULL;
	cudaMalloc((void**)&cudagrid, int(gridSize.x*gridSize.y*gridSize.z)*sizeof(pair<int,int>));
}

void findNeighbors(int numParticles, int maxNeighbors, glm::vec3 gridSize, float h){
	dim3 threadsPerBlock(64);
	dim3 fullBlocksPerGrid(numParticles/8+1);

	dim3 threadsPerBlockGrid(64);
	dim3 fullBlocksPerGridGrid(int(gridSize.x*gridSize.y*gridSize.z)/8+1);

	hashParticlesToGridKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(numParticles, cudacellIds, cudapIds, cudapositions, gridSize, cudaids, h);
	cudaThreadSynchronize();
	checkCUDAError("hasing particles");

	thrust::device_ptr<int> thrustcellIds = thrust::device_pointer_cast(cudacellIds);
	thrust::device_ptr<int> thrustpIds = thrust::device_pointer_cast(cudapIds);
	thrust::sort_by_key(thrustcellIds, thrustcellIds+numParticles, thrustpIds);

	resetGrid<<<fullBlocksPerGridGrid, threadsPerBlockGrid>>>(int(gridSize.x*gridSize.y*gridSize.z), cudagrid);
	checkCUDAError("reset grid");
	setGridValuesKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(numParticles, int(gridSize.x*gridSize.y*gridSize.z), cudagrid, cudacellIds);
	checkCUDAError("set values in grid neighbor");
	
	findNeighborsUsingGridKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(numParticles, h, maxNeighbors, cudapositions, cudaneighbors, cudanumNeighbors, cudagrid, cudapIds, int(gridSize.x*gridSize.y*gridSize.z), cudaids, gridSize, cudacellIds);
	checkCUDAError(" finding neighbors using grid ");
}

void freeCuda(){
	cudaFree(cudapositions);
	cudaFree(cudacellIds);
	cudaFree(cudapIds);
	cudaFree(cudagrid);
	cudaFree(cudanumNeighbors);
	cudaFree(cudaneighbors);
	cudaFree(cudaids);
}

void test_uniform_grid(){

	int * a = new int[10];
	int * b = new int[10];

	for (int i=0; i<10; i+=1){
		a[i] = i;
		b[i] = i;
	}
	
	dim3 threadsPerBlock(16);
	dim3 fullBlocksPerGrid(64);

	int *cudaA, *cudaB;

	cudaMalloc((void**)&cudaA, 10*sizeof(int));
	cudaMalloc((void**)&cudaB, 10*sizeof(int));

	cudaMemcpy(cudaA, a, 10*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, b, 10*sizeof(int), cudaMemcpyHostToDevice);

	hello<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaA, cudaB);

	cudaMemcpy(a, cudaA, 10*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, cudaB, 10*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i=0; i<10; i+=1){
		cout<<a[i]<<endl;
	}

	cudaFree(cudaA);
	cudaFree(cudaB);

	delete [] a;
	delete [] b;

}