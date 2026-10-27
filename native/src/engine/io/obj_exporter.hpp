#pragma once

#include "engine/io/mesh.hpp"
#include <filesystem>
#include <string>

namespace rhizome {

struct ManifoldCheckResult {
    bool is_manifold = true;
    bool is_watertight = true;
    size_t boundary_edges = 0;
    size_t non_manifold_edges = 0;
    size_t self_intersections = 0;
    std::string report;
};

class ObjExporter {
public:
    /**
     * @brief Validates 2-manifold topological integrity and watertightness.
     * Checks that:
     *  - Every edge is shared by exactly 2 faces.
     *  - Faces sharing an edge have opposite topological traversal (consistent outward orientation).
     *  - No edge is shared by 3 or more faces (non-manifold edge).
     *  - No disconnected boundary loops (boundary_edges == 0).
     */
    static ManifoldCheckResult verify_manifold(const TriangleMesh& mesh);

    /**
     * @brief Exports TriangleMesh to standard Wavefront OBJ format.
     * - Units: 1 OBJ unit = 1 millimeter.
     * - ASCII with fixed 6-decimal precision (e.g. "v 12.345678 0.123456 -5.000000").
     * - Enforces manifold verification prior to writing.
     */
    static bool export_obj(
        const std::filesystem::path& file_path,
        const TriangleMesh& mesh,
        std::string& out_error,
        bool enforce_manifold_check = true
    );

    static bool export_binary_stl(
        const std::filesystem::path& file_path,
        const TriangleMesh& mesh,
        std::string& out_error,
        bool enforce_manifold_check = true
    );

    static bool export_binary_ply(
        const std::filesystem::path& file_path,
        const TriangleMesh& mesh,
        std::string& out_error,
        bool enforce_manifold_check = true
    );

    static bool export_mesh(
        const std::filesystem::path& file_path,
        const TriangleMesh& mesh
    );
};

} // namespace rhizome
