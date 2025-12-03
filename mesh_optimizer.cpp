#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>

struct Vertex {
    float x, y, z;
    float r, g, b;
    
    Vertex(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z), r(1), g(1), b(1) {}
    
    float distance(const Vertex& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

struct Triangle {
    int v1, v2, v3;
    
    Triangle(int v1, int v2, int v3) : v1(v1), v2(v2), v3(v3) {}
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
};

// Simple PLY reader
bool readPLY(const std::string& filename, Mesh& mesh) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int vertex_count = 0;
    int face_count = 0;
    bool has_color = false;
    
    // Parse header
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &vertex_count);
        }
        else if (line.find("element face") != std::string::npos) {
            sscanf(line.c_str(), "element face %d", &face_count);
        }
        else if (line.find("property uchar red") != std::string::npos) {
            has_color = true;
        }
        else if (line == "end_header") {
            break;
        }
    }
    
    // Read vertices
    mesh.vertices.reserve(vertex_count);
    for (int i = 0; i < vertex_count; i++) {
        Vertex v;
        if (has_color) {
            int r, g, b;
            file >> v.x >> v.y >> v.z >> r >> g >> b;
            v.r = r / 255.0f;
            v.g = g / 255.0f;
            v.b = b / 255.0f;
        } else {
            file >> v.x >> v.y >> v.z;
        }
        mesh.vertices.push_back(v);
    }
    
    // Read faces
    mesh.triangles.reserve(face_count);
    for (int i = 0; i < face_count; i++) {
        int count, v1, v2, v3;
        file >> count >> v1 >> v2 >> v3;
        if (count == 3) {
            mesh.triangles.push_back(Triangle(v1, v2, v3));
        }
    }
    
    file.close();
    return true;
}

// Simple PLY writer
bool writePLY(const std::string& filename, const Mesh& mesh) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to create: " << filename << std::endl;
        return false;
    }
    
    // Write header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << mesh.vertices.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << mesh.triangles.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    
    // Write vertices
    for (const auto& v : mesh.vertices) {
        file << v.x << " " << v.y << " " << v.z << " "
             << (int)(v.r * 255) << " " << (int)(v.g * 255) << " " << (int)(v.b * 255) << "\n";
    }
    
    // Write faces
    for (const auto& t : mesh.triangles) {
        file << "3 " << t.v1 << " " << t.v2 << " " << t.v3 << "\n";
    }
    
    file.close();
    return true;
}

// Remove duplicate vertices
void removeDuplicates(Mesh& mesh, float threshold = 0.01f) {
    std::vector<Vertex> newVertices;
    std::vector<int> vertexMap(mesh.vertices.size());
    
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        bool found = false;
        for (size_t j = 0; j < newVertices.size(); j++) {
            if (mesh.vertices[i].distance(newVertices[j]) < threshold) {
                vertexMap[i] = j;
                found = true;
                break;
            }
        }
        
        if (!found) {
            vertexMap[i] = newVertices.size();
            newVertices.push_back(mesh.vertices[i]);
        }
    }
    
    // Update triangle indices
    for (auto& tri : mesh.triangles) {
        tri.v1 = vertexMap[tri.v1];
        tri.v2 = vertexMap[tri.v2];
        tri.v3 = vertexMap[tri.v3];
    }
    
    mesh.vertices = newVertices;
    
    std::cout << "Vertices reduced from " << vertexMap.size() << " to " << newVertices.size() << std::endl;
}

// Remove degenerate triangles
void removeInvalidTriangles(Mesh& mesh) {
    std::vector<Triangle> validTriangles;
    
    for (const auto& tri : mesh.triangles) {
        if (tri.v1 != tri.v2 && tri.v2 != tri.v3 && tri.v1 != tri.v3) {
            if (tri.v1 < (int)mesh.vertices.size() && 
                tri.v2 < (int)mesh.vertices.size() && 
                tri.v3 < (int)mesh.vertices.size()) {
                validTriangles.push_back(tri);
            }
        }
    }
    
    std::cout << "Triangles reduced from " << mesh.triangles.size() 
              << " to " << validTriangles.size() << std::endl;
    
    mesh.triangles = validTriangles;
}

// Smooth mesh (Laplacian smoothing)
void smoothMesh(Mesh& mesh, int iterations = 3) {
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<Vertex> newVertices = mesh.vertices;
        std::vector<int> neighborCount(mesh.vertices.size(), 0);
        std::vector<Vertex> neighborSum(mesh.vertices.size());
        
        // Calculate neighbor sums
        for (const auto& tri : mesh.triangles) {
            neighborSum[tri.v1].x += mesh.vertices[tri.v2].x + mesh.vertices[tri.v3].x;
            neighborSum[tri.v1].y += mesh.vertices[tri.v2].y + mesh.vertices[tri.v3].y;
            neighborSum[tri.v1].z += mesh.vertices[tri.v2].z + mesh.vertices[tri.v3].z;
            neighborCount[tri.v1] += 2;
            
            neighborSum[tri.v2].x += mesh.vertices[tri.v1].x + mesh.vertices[tri.v3].x;
            neighborSum[tri.v2].y += mesh.vertices[tri.v1].y + mesh.vertices[tri.v3].y;
            neighborSum[tri.v2].z += mesh.vertices[tri.v1].z + mesh.vertices[tri.v3].z;
            neighborCount[tri.v2] += 2;
            
            neighborSum[tri.v3].x += mesh.vertices[tri.v1].x + mesh.vertices[tri.v2].x;
            neighborSum[tri.v3].y += mesh.vertices[tri.v1].y + mesh.vertices[tri.v2].y;
            neighborSum[tri.v3].z += mesh.vertices[tri.v1].z + mesh.vertices[tri.v2].z;
            neighborCount[tri.v3] += 2;
        }
        
        // Apply smoothing
        for (size_t i = 0; i < mesh.vertices.size(); i++) {
            if (neighborCount[i] > 0) {
                float lambda = 0.5f;
                newVertices[i].x = mesh.vertices[i].x * (1 - lambda) + 
                                   (neighborSum[i].x / neighborCount[i]) * lambda;
                newVertices[i].y = mesh.vertices[i].y * (1 - lambda) + 
                                   (neighborSum[i].y / neighborCount[i]) * lambda;
                newVertices[i].z = mesh.vertices[i].z * (1 - lambda) + 
                                   (neighborSum[i].z / neighborCount[i]) * lambda;
            }
        }
        
        mesh.vertices = newVertices;
    }
    
    std::cout << "Applied " << iterations << " smoothing iterations" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.ply> <output.ply>" << std::endl;
        return 1;
    }
    
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    
    std::cout << "Reading mesh from: " << inputFile << std::endl;
    
    Mesh mesh;
    if (!readPLY(inputFile, mesh)) {
        return 1;
    }
    
    std::cout << "Initial mesh: " << mesh.vertices.size() << " vertices, " 
              << mesh.triangles.size() << " triangles" << std::endl;
    
    // Optimize
    removeDuplicates(mesh, 0.01f);
    removeInvalidTriangles(mesh);
    smoothMesh(mesh, 2);
    
    std::cout << "Final mesh: " << mesh.vertices.size() << " vertices, " 
              << mesh.triangles.size() << " triangles" << std::endl;
    
    std::cout << "Writing optimized mesh to: " << outputFile << std::endl;
    
    if (!writePLY(outputFile, mesh)) {
        return 1;
    }
    
    std::cout << "Optimization complete!" << std::endl;
    return 0;
}
