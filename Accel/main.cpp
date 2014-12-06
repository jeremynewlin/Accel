#include "../glew/include/GL/glew.h"
#include "../glfw/include/GL/glfw.h"

#include <iostream>
#include <time.h>

#include "uniform_grid.h"
#include "mesh.h"
#include "KDTreeCPU.h"

using namespace std;

int windowWidth  = 750;
int windowHeight = 750;

glm::vec3 upperBound = glm::vec3(1,1,1);
glm::vec3 lowerBound = glm::vec3(0,0,0);

glm::vec3 cameraPos(0.5,0.5,2);
glm::vec3 cameraTar(0.5,0.5,0);

glm::vec3 cameraUp(0,1,0);

glm::vec3 lightPos(0,10,0);

bool paused   = false;
bool drawGridToggle = false, drawHashToggle = false;
bool printDistances = true;

int currentID = 0;
int numIDs = 1;

std::vector<glm::vec3> colors;

void aimCamera(){
	gluPerspective(45, static_cast<float>(windowWidth) / static_cast<float>(windowHeight), 0.1, 10000.0);

	gluLookAt(cameraPos.x, cameraPos.y, cameraPos.z,
			  cameraTar.x, cameraTar.y, cameraTar.z,
			   cameraUp.x,  cameraUp.y,  cameraUp.z);
}

void drawGrid(glm::vec3 gridSize, float h, glm::vec3 offset){
	int gx, gy,gz;
	gx = int(gridSize.x);
	gy = int(gridSize.y);
	gz = int(gridSize.z);
	float cellSize = h;
	
	for (int i=0; i<gx; i++){
		for (int j=0; j<gy; j++){
			for (int k=0; k<gz; k++){
				glm::vec3 min = glm::vec3(i*cellSize,j*cellSize,k*cellSize) + offset;
				glm::vec3 max = min+glm::vec3(cellSize,cellSize,cellSize);
				glColor4f(1,1,1, 0.25f);
				glBegin(GL_LINE_LOOP); //top
				glVertex3f(min.x,max.y,min.z);
				glVertex3f(max.x,max.y,min.z);
				glVertex3f(max.x,max.y,max.z);
				glVertex3f(min.x,max.y,max.z);
				glEnd();

				glBegin(GL_LINE_LOOP); //bottom
				glVertex3f(min.x,min.y,min.z);
				glVertex3f(max.x,min.y,min.z);
				glVertex3f(max.x,min.y,max.z);
				glVertex3f(min.x,min.y,max.z);
				glEnd();

				glBegin(GL_LINE_LOOP); //right
				glVertex3f(max.x,min.y,min.z);
				glVertex3f(max.x,max.y,min.z);
				glVertex3f(max.x,max.y,max.z);
				glVertex3f(max.x,min.y,max.z);
				glEnd();

				glBegin(GL_LINE_LOOP); //left
				glVertex3f(min.x,min.y,min.z);
				glVertex3f(min.x,max.y,min.z);
				glVertex3f(min.x,max.y,max.z);
				glVertex3f(min.x,min.y,max.z);
				glEnd();
			}
		}
	}
}

void drawMesh(mesh* m){
	int numTris = m->numTris;
	for (int i=0; i<numTris; i+=1){

		glm::vec3 tri = m->tris[i];

		int ix = int(tri.x);
		int iy = int(tri.y);
		int iz = int(tri.z);

		glm::vec3 p1 = m->verts[ix];
		glm::vec3 p2 = m->verts[iy];
		glm::vec3 p3 = m->verts[iz];

		glm::vec3 normal = glm::cross(p2-p1, p1-p3);
		normal = glm::abs(glm::normalize(normal));

		glColor3f(normal.x, normal.y, normal.z);
		glBegin(GL_TRIANGLES); //top
		glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
		glVertex3f(p3.x,p3.y,p3.z);
		glEnd();

	}
}

void drawMeshAsPoints(mesh* m){
	int numPoints = m->numVerts;
	for (int i=0; i<numPoints; i+=1){

		glm::vec3 p1 = m->verts[i];

		glColor4f(1,1,1, 0.25f);
		glBegin(GL_POINTS); //top
		glVertex3f(p1.x,p1.y,p1.z);
		glEnd();

	}
}

void drawNeighbors(int particleID, const hash_grid& grid, bool useGrid = true){
	int numPoints = grid.m_gridNumNeighbors[particleID];
	if (!useGrid) numPoints = grid.m_bruteNumNeighbors[particleID];

	for (int i=0; i<numPoints; i+=1){

		int id;
		if (useGrid) id = grid.m_gridNeighbors[particleID*grid.m_maxNeighbors + i];
		else id = grid.m_bruteNeighbors[particleID*grid.m_maxNeighbors + i];
		
		glm::vec3 p1 = grid.m_points[id];

		glColor4f(1,0,0, 0.75f);
		if (!useGrid) glColor4f(0,0,1, 0.75f);

		glBegin(GL_POINTS); //top
		glVertex3f(p1.x,p1.y,p1.z);
		glEnd();
	}

	glm::vec3 p1 = grid.m_points[particleID];
	glColor3f(0,1,0);
	glBegin(GL_POINTS); //top
	glVertex3f(p1.x,p1.y,p1.z);
	glEnd();
}

void drawHashes(const hash_grid& grid){
	int numPoints = grid.m_numParticles;
	for (int i=0; i<numPoints; i+=1){

		glm::vec3 p1 = grid.m_points[i];
		int hash = grid.hashParticle(i);
		glm::vec3 c1 = colors[hash];

		glColor4f(c1.x,c1.y,c1.z, 0.25f);
		glBegin(GL_POINTS); //top
		glVertex3f(p1.x,p1.y,p1.z);
		glEnd();

	}
}

void drawBoundingBox( boundingBox bb, glm::vec3 col=glm::vec3( 1.0f, 1.0f, 1.0f ) )
{
	glm::vec3 min = bb.min;
	glm::vec3 max = bb.max;

	glColor3f( col.x, col.y, col.z );
	glBegin(GL_LINE_LOOP); //top
	glVertex3f(min.x,max.y,min.z);
	glVertex3f(max.x,max.y,min.z);
	glVertex3f(max.x,max.y,max.z);
	glVertex3f(min.x,max.y,max.z);
	glEnd();

	glBegin(GL_LINE_LOOP); //bottom
	glVertex3f(min.x,min.y,min.z);
	glVertex3f(max.x,min.y,min.z);
	glVertex3f(max.x,min.y,max.z);
	glVertex3f(min.x,min.y,max.z);
	glEnd();

	glBegin(GL_LINE_LOOP); //right
	glVertex3f(max.x,min.y,min.z);
	glVertex3f(max.x,max.y,min.z);
	glVertex3f(max.x,max.y,max.z);
	glVertex3f(max.x,min.y,max.z);
	glEnd();

	glBegin(GL_LINE_LOOP); //left
	glVertex3f(min.x,min.y,min.z);
	glVertex3f(min.x,max.y,min.z);
	glVertex3f(min.x,max.y,max.z);
	glVertex3f(min.x,min.y,max.z);
	glEnd();
}

void keypress(int key, int action)
{
	if(glfwGetKey(key) == GLFW_PRESS)
	{
		switch(key)
		{
			case GLFW_KEY_SPACE:
				break;
			case GLFW_KEY_ESC:
				break;
			case 'a':
			case 'A':
				currentID = (currentID-1);//%numIDs;
				if (currentID < 0) currentID = numIDs-1;
				currentID = currentID % numIDs;
				printDistances = true;
				break;
			case 'h':
			case 'H':
				drawHashToggle = !drawHashToggle;
				break;
			case 'v':
			case 'V':
				break;
			case 'q':
			case 'Q':
				break;
			case 'w':
			case 'W':
				break;
			case 'p':
			case 'P':
				paused = !paused;
				break;
			case 'r':
			case 'R':
				break;
			case 'g':
			case 'G':
				drawGridToggle=!drawGridToggle;
				break;
			case 'd':
			case 'D':
				break;
			case 'b':
			case 'B':
				break;
			case 's':
			case 'S':
				currentID = (currentID+1)%numIDs;
				printDistances = true;
				break;
			case 'z':
			case 'Z':
				break;
		}
	}
}

void getBB(KDTreeNode* current, vector<boundingBox>& bbs){

}

// Draw kd-tree bounding boxes.
void drawKDTree( KDTreeNode *curr_node, int curr_depth, int max_num_levels )
{
	glm::vec3 base_col( 1.0f, 1.0f, 1.0f );

	float strength = ( ( float )curr_depth / max_num_levels );
	glm::vec3 col = glm::vec3( base_col.x * strength, base_col.y * strength, base_col.z * strength );
	drawBoundingBox( curr_node->bbox, col );

	if ( curr_node->left ) {
		drawKDTree( curr_node->left, curr_depth + 1, max_num_levels );
	}
	if ( curr_node->right ) {
		drawKDTree( curr_node->right, curr_depth + 1, max_num_levels );
	}
}

void runTimingComparison(hash_grid& grid, float h){
	int nums[6] = {10, 25, 50, 100, 250, 500};
	for (int i=0; i<6; i+=1){
		bool useGPU = true;
		bool useGrid = false;

		clock_t t = clock();
		//grid.findNeighbors(nums[i], h, useGrid, useGPU);
		grid.findNeighbors(nums[i], h, useGrid, useGPU);
		t = clock() - t;

		cout<<"for "<<grid.m_maxNeighbors<<" neighbors, using ";
		if (useGPU) cout<<"gpu ";
		else cout<<"cpu ";

		if (useGrid) cout<<"grid, ";
		else cout<<"brute force, ";

		cout<<"the nearest neighbor search took ";
		cout<<((float)t)/CLOCKS_PER_SEC;
		cout<<" seconds"<<endl;
	}
}

int runKD(){
	srand(time(NULL));

	// Initialize kd-tree.
	mesh* m = new mesh( "meshes\\bunny_small.obj" );
	KDTreeCPU kd_tree = KDTreeCPU( m->numTris, m->tris, m->numVerts, m->verts );

	bool run = GL_TRUE;

    if(!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    if(!glfwOpenWindow(static_cast<int>(windowWidth), static_cast<int>(windowHeight), 8, 8, 8, 8, 24, 0, GLFW_WINDOW))
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glewInit();
    if (!glewIsSupported( "GL_VERSION_2_0 " "GL_ARB_pixel_buffer_object")) {
            fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
            fflush( stderr);
            return false;
    }

    glfwSetKeyCallback(keypress);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glViewport(0, 0, static_cast<GLsizei>(windowWidth), static_cast<GLsizei>(windowHeight));
	glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glPointSize( 6.0 );

	aimCamera();

	int frame=0;
	float lastTime = glfwGetTime();
	while(run){
		frame+=1;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Visualize kd-tree.
		drawMesh( m );
		drawKDTree( kd_tree.getRootNode(), 1, kd_tree.getNumLevels() );

		GLenum errCode;
		const GLubyte* errString;
		if (errCode=glGetError() != GL_NO_ERROR){
			glfwTerminate();
			exit(1);
		}

		if (paused){
			glfwSetWindowTitle("Paused");
		}
		else{
			float now = glfwGetTime();
			char fpsInfo[256];
			sprintf(fpsInfo, "Accel Library Visual Testing | Framerate: %f", 1.0f / (now - lastTime));
			lastTime = now;
			glfwSetWindowTitle(fpsInfo);
		}

		glfwSwapBuffers();
		run = !glfwGetKey(GLFW_KEY_ESC) && glfwGetWindowParam(GLFW_OPENED);
	}

	delete m;

	glfwTerminate();
    exit(EXIT_SUCCESS);
}

int runGrid(){
	srand(time(NULL));

	

	for (int i=0; i<10000; i+=1){
		float x = float(rand())/float(RAND_MAX);
		float y = float(rand())/float(RAND_MAX);
		float z = float(rand())/float(RAND_MAX);
		colors.push_back(glm::vec3(x,y,z));
	}

	mesh* m = new mesh("meshes\\bunny_small_1.obj");

	glm::vec3 gridSize = m->bb.max - m->bb.min;
	gridSize = glm::vec3(1,1,1);
	
	
	float h = 0.01f;


	gridSize /= h;

	gridSize.x = floor(gridSize.x)+1.0f;
	gridSize.y = floor(gridSize.y)+1.0f;
	gridSize.z = floor(gridSize.z)+1.0f;

	
	cout<<(gridSize.x*gridSize.x*gridSize.x)*sizeof(int)<<endl;
	cout<<sizeof(int)<<endl;

	hash_grid grid = hash_grid(m->numVerts, m->verts, gridSize);
	runTimingComparison(grid, h);
	//grid.findNeighbors(250, h, true, true);
	//grid.findNeighbors(250, h, false, true);
	//grid.findNeighbors(250, h, true, true);

	numIDs = grid.m_numParticles;

	bool run = GL_TRUE;

    if(!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    if(!glfwOpenWindow(static_cast<int>(windowWidth), static_cast<int>(windowHeight), 8, 8, 8, 8, 24, 0, GLFW_WINDOW))
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glewInit();
    if (!glewIsSupported( "GL_VERSION_2_0 " "GL_ARB_pixel_buffer_object")) {
            fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
            fflush( stderr);
            return false;
    }

    glfwSetKeyCallback(keypress);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glViewport(0, 0, static_cast<GLsizei>(windowWidth), static_cast<GLsizei>(windowHeight));
	glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glPointSize( 6.0 );

	aimCamera();

	int frame=0;
	float lastTime = glfwGetTime();
	while(run){
		frame+=1;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//if (printDistances){
		//	float avgDist = 0;
		//	for (int i=0; i<grid.m_bruteNumNeighbors[currentID]; i+=1){
		//		avgDist += glm::distance(grid.m_points[currentID], grid.m_points[grid.m_bruteNeighbors[currentID*grid.m_maxNeighbors + i]]);
		//	}
		//	cout<<"average from brute force: "<<avgDist/grid.m_bruteNumNeighbors[currentID]<<endl;

		//	float bruteAvg = avgDist;

		//	avgDist = 0;
		//	for (int i=0; i<grid.m_gridNumNeighbors[currentID]; i+=1){
		//		avgDist += glm::distance(grid.m_points[currentID], grid.m_points[grid.m_gridNeighbors[currentID*grid.m_maxNeighbors + i]]);
		//	}
		//	cout<<"average from grid       : "<<avgDist/grid.m_gridNumNeighbors[currentID]<<endl<<endl;
		//	printDistances = false;
		//}
		if (drawGridToggle) drawGrid(grid.m_gridSize, h, glm::vec3()/*-(m->bb.max-m->bb.min)/2.0f*/);
		drawMeshAsPoints(m);
		if (drawHashToggle) drawHashes(grid);
		drawNeighbors(currentID, grid, false);
		drawNeighbors(currentID, grid, true);

		GLenum errCode;
		const GLubyte* errString;
		if (errCode=glGetError() != GL_NO_ERROR){
			glfwTerminate();
			exit(1);
		}

		if (paused){
			glfwSetWindowTitle("Paused");
		}
		else{
			float now = glfwGetTime();
			char fpsInfo[256];
			sprintf(fpsInfo, "Accel Library Visual Testing | Framerate: %f", 1.0f / (now - lastTime));
			lastTime = now;
			glfwSetWindowTitle(fpsInfo);
		}

		glfwSwapBuffers();
		run = !glfwGetKey(GLFW_KEY_ESC) && glfwGetWindowParam(GLFW_OPENED);
	}

	delete m;

	glfwTerminate();
    exit(EXIT_SUCCESS);
}

int main(){

	//return runGrid();
	return runKD();
}