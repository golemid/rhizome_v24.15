#pragma once

#include "engine/io/mesh.hpp"
#include <filesystem>
#include <string>

namespace rhizome {

struct WindingPrepassResult {
    bool is_orientable = true;
    bool had_inconsistent_winding = false;
    bool was_inverted = false;
    float signed_volume = 0.0f;
    std::string message;
};

class MeshWindingPass {
public:
    /**
     * @brief Normal-consistency BFS propagation and signed-volume orientation pre-pass.
     * Unifies face winding across manifold edges and flips negatively signed volumes outward.
     * Flags non-orientable topologies without silent failure.
     */
    static WindingPrepassResult orient_mesh(TriangleMesh& mesh);
};

class ObjParser {
public:
    /**
     * @brief Parse Wavefront OBJ file into flat vertex and index arrays.
     * Supports triangulated polygonal faces (quads/polygons are fan-triangulated).
     * Applies orientation pre-pass automatically.
     */
    static bool parse(const std::filesystem::path& file_path, TriangleMesh& out_mesh, std::string& out_error);
};

class StlParser {
public:
    /**
     * @brief Parse both Binary and ASCII STL files into indexed flat vertex and index arrays.
     * Automatically welds coincident vertices within geometric tolerance (1e-5 mm).
     * Applies orientation pre-pass automatically.
     */
    static bool parse(const std::filesystem::path& file_path, TriangleMesh& out_mesh, std::string& out_error);

    static bool parse_with_normals(
        const std::filesystem::path& file_path,
        TriangleMesh& out_mesh,
        std::vector<float>& out_normals,
        std::string& out_error
    );
};

} // namespace rhizome
