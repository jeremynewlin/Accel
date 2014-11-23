#include "mesh.h"

using namespace std;

mesh::mesh(){
	numTris = numVerts = 0;
}

mesh::mesh(string fileName){
	ifstream ifile;
	string line;
	ifile.open(fileName.c_str());

	vector<glm::vec3> verts, faces;

	while (utilityCore::safeGetline(ifile, line)) {
		vector<string> tokens = utilityCore::tokenizeString(line);

		if (tokens.size()>0 && strcmp(tokens[0].c_str(),"v")==0){
			verts.push_back(glm::vec3(atof(tokens[1].c_str()),atof(tokens[2].c_str()),atof(tokens[3].c_str())));
		}
		else if (tokens.size()>0 && strcmp(tokens[0].c_str(),"f")==0){
			char* findex1 = strtok (const_cast<char*>(tokens[1].c_str()),"/");
			char* findex2 = strtok (const_cast<char*>(tokens[2].c_str()),"/");
			char* findex3 = strtok (const_cast<char*>(tokens[3].c_str()),"/");
			faces.push_back(glm::vec3(atof(findex1)-1,atof(findex2)-1,atof(findex3)-1));
		}
	}

	glm::vec3* vertData = new glm::vec3[verts.size()];
	glm::vec3* faceData = new glm::vec3[faces.size()];

	glm::vec3 max = glm::vec3(-10000,-10000,-10000);
	glm::vec3 min = glm::vec3( 10000, 10000, 10000);

	for (int i=0; i<verts.size(); i++){
		vertData[i] = verts[i];
		if (verts[i].x<min.x){
			min.x=verts[i].x;
		}
		if (verts[i].y<min.y){
			min.y=verts[i].y;
		}
		if (verts[i].z<min.z){
			min.z=verts[i].z;
		}
		if (verts[i].x>max.x){
			max.x=verts[i].x;
		}
		if (verts[i].y>max.y){
			max.y=verts[i].y;
		}
		if (verts[i].z>max.z){
			max.z=verts[i].z;
		}
	}
	for (int i=0; i<faces.size(); i++){
		faceData[i] = faces[i];
	}

	this->tris = faceData;
	this->verts = vertData;
	this->numTris = faces.size();
	this->numVerts = verts.size();

	bb.min = min;
	bb.max = max;
}