#include "uniform_grid.h"

using namespace std;

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
void findNeighborsUsingGridKernel(int numParticles, float h, int maxNeighbors, glm::vec3* predictions, int* neighbors, int* numNeighbors, pair<int,int>* grid, int* pIds, int numGridCells, int* ids, glm::vec3 gridSize, int* cellIds){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index<numParticles){
		index = ids[index];
		numNeighbors[index] = 0;

		int cellIdOriginal = hashParticle(predictions[index], gridSize, h);
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

			if (cellId<0 || cellId>numGridCells-1) continue;
			cellId1 = grid[cellId].first;
			if (cellId1==-1) continue;
			cellId2 = grid[cellId].second;
			if (cellId2==-1) cellId2=cellId1;
			if (cellId1>numParticles-1 || cellId1<0 || cellId2>numParticles-1 || cellId2<0) continue;

			int jid = ids[pIds[cellId1]];
			if (lengthSquared(predictions[index]-predictions[jid])<h*h && numNeighbors[index]<maxNeighbors){
				int nid = index*maxNeighbors+numNeighbors[index];
				neighbors[nid]=jid;
				numNeighbors[index]+=1;
			}

			for (int i=cellId1; i<cellId2+1; i+=1){
				int jid = ids[pIds[i]];
				if (lengthSquared(predictions[index]-predictions[jid])<h*h && numNeighbors[index]<maxNeighbors){
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