#include "engine/io/parsers.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include <fstream>
#include <iostream>

namespace rhizome {

WindingPrepassResult MeshWindingPass::orient_mesh(TriangleMesh& mesh) {
    WindingPrepassResult result;
    const size_t num_tris = mesh.triangle_count();
    if (num_tris == 0) return result;

    float gpu_time_ms = 0.0f;
    CudaRasterizer::orient_mesh_gpu(
        mesh,
        result.is_orientable,
        result.had_inconsistent_winding,
        result.was_inverted,
        result.signed_volume,
        gpu_time_ms
    );

    if (!result.is_orientable) {
        result.message = "Mesh contains non-orientable topology (e.g. Mobius band).";
    } else if (result.was_inverted) {
        result.message = "Mesh normals were inverted; corrected to outward orientation.";
    } else if (result.had_inconsistent_winding) {
        result.message = "Inconsistent face winding detected and unified.";
    } else {
        result.message = "Mesh orientation consistent and outward-facing.";
    }

    return result;
}

bool ObjParser::parse(const std::filesystem::path& file_path, TriangleMesh& out_mesh, std::string& out_error) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        out_error = "Failed to open OBJ file: " + file_path.string();
        return false;
    }

    file.seekg(0, std::ios::end);
    const auto stream_pos = file.tellg();
    if (stream_pos <= 0) {
        out_error = "OBJ file is empty or cannot determine size";
        return false;
    }
    const size_t file_size = static_cast<size_t>(stream_pos);
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    file.read(buffer.data(), file_size);

    float gpu_time_ms = 0.0f;
    if (CudaRasterizer::parse_obj_text_gpu(buffer.data(), buffer.size(), out_mesh, gpu_time_ms)) {
        auto orient_res = MeshWindingPass::orient_mesh(out_mesh);
        if (!orient_res.is_orientable) {
            out_error = "[Warning] " + orient_res.message;
        }
        return true;
    }

    out_error = "GPU parallel ASCII OBJ decoding failed";
    return false;
}

} // namespace rhizome
