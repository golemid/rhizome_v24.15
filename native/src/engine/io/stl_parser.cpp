#include "engine/io/parsers.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include <fstream>
#include <vector>
#include <cstring>

namespace rhizome {

bool StlParser::parse(const std::filesystem::path& file_path, TriangleMesh& out_mesh, std::string& out_error) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        out_error = "Failed to open STL file: " + file_path.string();
        return false;
    }

    // Determine file size safely
    file.seekg(0, std::ios::end);
    const auto stream_pos = file.tellg();
    if (stream_pos < 0) {
        out_error = "Failed to determine STL file size";
        return false;
    }
    const size_t file_size = static_cast<size_t>(stream_pos);
    file.seekg(0, std::ios::beg);

    if (file_size < 84) {
        out_error = "STL file too small to be valid";
        return false;
    }

    out_mesh.vertices.clear();
    out_mesh.indices.clear();

    // Single-pass buffer load directly from disk into memory
    std::vector<uint8_t> file_buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(file_buffer.data()), file_size)) {
        out_error = "Failed to read STL file data";
        return false;
    }

    uint32_t num_triangles = 0;
    std::memcpy(&num_triangles, file_buffer.data() + 80, sizeof(uint32_t));

    const bool is_binary = (num_triangles > 0 && file_size >= (84ULL + static_cast<size_t>(num_triangles) * 50ULL));

    float gpu_time_ms = 0.0f;
    if (is_binary) {
        if (!CudaRasterizer::parse_binary_stl_gpu(file_buffer.data(), file_buffer.size(), out_mesh, gpu_time_ms)) {
            out_error = "GPU binary STL parallel decoding failed";
            return false;
        }
    } else {
        if (!CudaRasterizer::parse_ascii_stl_gpu(reinterpret_cast<const char*>(file_buffer.data()), file_buffer.size(), out_mesh, gpu_time_ms)) {
            out_error = "GPU ASCII STL parallel decoding failed";
            return false;
        }
    }

    if (out_mesh.indices.empty()) {
        out_error = "STL contains no valid non-degenerate triangles";
        return false;
    }

    auto orient_res = MeshWindingPass::orient_mesh(out_mesh);
    if (!orient_res.is_orientable) {
        out_error = "[Warning] " + orient_res.message;
    }

    return true;
}

bool StlParser::parse_with_normals(
    const std::filesystem::path& file_path,
    TriangleMesh& out_mesh,
    std::vector<float>& out_normals,
    std::string& out_error
) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        out_error = "Failed to open STL file: " + file_path.string();
        return false;
    }

    // Determine file size safely
    file.seekg(0, std::ios::end);
    const auto stream_pos = file.tellg();
    if (stream_pos < 0) {
        out_error = "Failed to determine STL file size";
        return false;
    }
    const size_t file_size = static_cast<size_t>(stream_pos);
    file.seekg(0, std::ios::beg);

    if (file_size < 84) {
        out_error = "STL file too small to be valid";
        return false;
    }

    std::vector<uint8_t> file_buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(file_buffer.data()), file_size)) {
        out_error = "Failed to read STL file data";
        return false;
    }

    uint32_t num_triangles = 0;
    std::memcpy(&num_triangles, file_buffer.data() + 80, sizeof(uint32_t));

    const bool is_binary = (num_triangles > 0 && file_size >= (84ULL + static_cast<size_t>(num_triangles) * 50ULL));
    if (!is_binary) {
        bool ok = parse(file_path, out_mesh, out_error);
        if (ok) {
            float dummy_time = 0.0f;
            CudaRasterizer::compute_vertex_normals_gpu(out_mesh, out_normals, dummy_time);
        }
        return ok;
    }

    float gpu_time_ms = 0.0f;
    if (CudaRasterizer::parse_binary_stl_gpu(file_buffer.data(), file_buffer.size(), out_mesh, gpu_time_ms, &out_normals)) {
        return true;
    }
    out_error = "GPU binary STL parallel decoding failed";
    return false;
}

} // namespace rhizome
