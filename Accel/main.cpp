#include "glew.h"
#include "glfw.h"

#include <iostream>

#include "uniform_grid.h"
#include "mesh.h"
#include "kdtree.h"

using namespace std;

int windowWidth  = 750;
int windowHeight = 750;

glm::vec3 upperBound = glm::vec3(1,1,1);
glm::vec3 lowerBound = glm::vec3(0,0,0);

glm::vec3 cameraPos(0,5,10);
glm::vec3 cameraTar(0,0,0);
glm::vec3 cameraUp(0,1,0);

glm::vec3 lightPos(0,10,0);

bool paused = false;

void aimCamera(){
	gluPerspective(45, static_cast<float>(windowWidth) / static_cast<float>(windowHeight), 0.1, 10000.0);

	gluLookAt(cameraPos.x, cameraPos.y, cameraPos.z,
			  cameraTar.x, cameraTar.y, cameraTar.z,
			   cameraUp.x,  cameraUp.y,  cameraUp.z);
}

void drawGrid(){
	float h = 0.50f;
	glColor3f(0,.7,.7);
	int gx, gy,gz;
	glm::vec3 gridSize = glm::vec3(1,1,1)/h;
	gx = int(gridSize.x);
	gy = int(gridSize.y);
	gz = int(gridSize.z);
	float cellSize = h;
	
	for (int i=0; i<gx; i++){
		for (int j=0; j<gy; j++){
			for (int k=0; k<gz; k++){
				glm::vec3 min = glm::vec3(i*cellSize,j*cellSize,k*cellSize);
				glm::vec3 max = min+glm::vec3(cellSize,cellSize,cellSize);
				glColor3f(1,1,1);
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

		glColor3f(1,0,0);
		glBegin(GL_POINTS); //top
		glVertex3f(p1.x,p1.y,p1.z);
		glEnd();

	}
}

void drawBoundingBox(boundingBox bb){
	glm::vec3 min = bb.min;
	glm::vec3 max = bb.max;

	glColor3f(1,1,1);
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

void keypress(int key, int action){
}

int main(){

	mesh* m = new mesh("meshes\\bunny.obj");

	glm::vec3 gridSize = m->bb.max - m->bb.min;
	
	/*int* ids = new int[m->numVerts];
	for (int i=0; i<m->numVerts; i+=1){
		ids[i] = i;
	}
	initCuda(m->numVerts, ids, m->verts, 25, gridSize);
	findNeighbors(m->numVerts, 25, gridSize, 1.5);
	freeCudaGrid();
	delete [] ids;*/

	hash_grid grid = hash_grid(m->numVerts, m->verts, gridSize);
	grid.findNeighbors(25, 1.5);

	//kdtree kd(m);
	//kd.construct();

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

		//drawGrid();
		drawMeshAsPoints(m);
		//drawMesh(m);
		drawBoundingBox(m->bb);
		//for (int i=0; i<kd.m_mesh->numTris; i+=1){
		//	drawBoundingBox(kd.boundingBoxes[i]);
		//}

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