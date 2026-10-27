#include "engine/io/obj_exporter.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace rhizome {

ManifoldCheckResult ObjExporter::verify_manifold(const TriangleMesh& mesh) {
    ManifoldCheckResult result;
    float time_ms = 0.0f;
    size_t boundary = 0, non_manifold = 0, conflicts = 0;

    if (CudaRasterizer::verify_manifold_gpu(mesh, boundary, non_manifold, conflicts, time_ms)) {
        result.boundary_edges = boundary;
        result.non_manifold_edges = non_manifold;
        result.self_intersections = conflicts;
        result.is_watertight = (result.boundary_edges == 0);
        result.is_manifold = (result.non_manifold_edges == 0 && result.is_watertight);

        std::ostringstream ss;
        ss << "Manifold Check (GPU " << time_ms << " ms): "
           << (result.is_manifold ? "PASSED (Watertight 2-Manifold)" : "FAILED")
           << " | Boundary edges: " << result.boundary_edges
           << " | Non-manifold edges: " << result.non_manifold_edges
           << " | Orientation conflicts: " << result.self_intersections;
        result.report = ss.str();
        return result;
    }

    result.is_manifold = false;
    result.is_watertight = false;
    result.report = "GPU edge manifold verification failed";
    return result;
}

bool ObjExporter::export_obj(
    const std::filesystem::path& file_path,
    const TriangleMesh& mesh,
    std::string& out_error,
    bool enforce_manifold_check
) {
    if (enforce_manifold_check) {
        ManifoldCheckResult check = verify_manifold(mesh);
        if (!check.is_manifold) {
            out_error = "Mesh export aborted: topological verification failed. " + check.report;
            return false;
        }
    }

    if (file_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(file_path.parent_path(), ec);
    }

    std::vector<char> gpu_formatted_buffer;
    float time_ms = 0.0f;
    if (CudaRasterizer::format_obj_gpu(mesh, gpu_formatted_buffer, time_ms)) {
        std::ofstream out(file_path, std::ios::binary);
        if (!out.is_open()) {
            out_error = "Could not open file for writing: " + file_path.string();
            return false;
        }
        out.write(gpu_formatted_buffer.data(), gpu_formatted_buffer.size());
        return true;
    }

    out_error = "GPU-accelerated file formatting failed";
    return false;
}

bool ObjExporter::export_binary_stl(
    const std::filesystem::path& file_path,
    const TriangleMesh& mesh,
    std::string& out_error,
    bool enforce_manifold_check
) {
    if (enforce_manifold_check) {
        ManifoldCheckResult check = verify_manifold(mesh);
        if (!check.is_manifold) {
            out_error = "Mesh export aborted: topological verification failed. " + check.report;
            return false;
        }
    }

    if (file_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(file_path.parent_path(), ec);
    }

    std::vector<uint8_t> gpu_buffer;
    float time_ms = 0.0f;
    if (CudaRasterizer::format_binary_stl_gpu(mesh, gpu_buffer, time_ms)) {
        std::ofstream out(file_path, std::ios::binary);
        if (!out.is_open()) {
            out_error = "Could not open file for writing: " + file_path.string();
            return false;
        }
        out.write(reinterpret_cast<const char*>(gpu_buffer.data()), gpu_buffer.size());
        return true;
    }

    out_error = "GPU-accelerated binary STL formatting failed";
    return false;
}

bool ObjExporter::export_binary_ply(
    const std::filesystem::path& file_path,
    const TriangleMesh& mesh,
    std::string& out_error,
    bool enforce_manifold_check
) {
    if (enforce_manifold_check) {
        ManifoldCheckResult check = verify_manifold(mesh);
        if (!check.is_manifold) {
            out_error = "Mesh export aborted: topological verification failed. " + check.report;
            return false;
        }
    }

    if (file_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(file_path.parent_path(), ec);
    }

    std::vector<uint8_t> gpu_buffer;
    float time_ms = 0.0f;
    if (CudaRasterizer::format_binary_ply_gpu(mesh, gpu_buffer, time_ms)) {
        std::ofstream out(file_path, std::ios::binary);
        if (!out.is_open()) {
            out_error = "Could not open file for writing: " + file_path.string();
            return false;
        }
        out.write(reinterpret_cast<const char*>(gpu_buffer.data()), gpu_buffer.size());
        return true;
    }

    out_error = "GPU-accelerated binary PLY formatting failed";
    return false;
}

bool ObjExporter::export_mesh(
    const std::filesystem::path& file_path,
    const TriangleMesh& mesh
) {
    std::string err;
    std::string ext = file_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)::tolower(c); });

    if (ext == ".stl") {
        return export_binary_stl(file_path, mesh, err, false);
    } else if (ext == ".ply") {
        return export_binary_ply(file_path, mesh, err, false);
    } else {
        return export_obj(file_path, mesh, err, false);
    }
}

} // namespace rhizome
