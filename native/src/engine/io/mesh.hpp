#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <glm/glm.hpp>

namespace rhizome {

/**
 * @brief Canonical flat representation of 3D triangular surface meshes.
 * Coordinates are expressed in millimeters (1 unit = 1 mm).
 */
struct TriangleMesh {
    // Flat coordinate array: [x0, y0, z0, x1, y1, z1, ...]
    std::vector<float> vertices;

    // Flat index array: [i0, i1, i2, i3, i4, i5, ...] (3 indices per face)
    std::vector<uint32_t> indices;

    [[nodiscard]] size_t vertex_count() const noexcept {
        return vertices.size() / 3;
    }

    [[nodiscard]] size_t triangle_count() const noexcept {
        return indices.size() / 3;
    }

    [[nodiscard]] glm::vec3 get_vertex(size_t index) const noexcept {
        const size_t offset = index * 3;
        return {vertices[offset], vertices[offset + 1], vertices[offset + 2]};
    }

    void set_vertex(size_t index, const glm::vec3& v) noexcept {
        const size_t offset = index * 3;
        vertices[offset] = v.x;
        vertices[offset + 1] = v.y;
        vertices[offset + 2] = v.z;
    }

    void get_triangle(size_t tri_idx, glm::vec3& v0, glm::vec3& v1, glm::vec3& v2) const noexcept {
        const size_t offset = tri_idx * 3;
        v0 = get_vertex(indices[offset]);
        v1 = get_vertex(indices[offset + 1]);
        v2 = get_vertex(indices[offset + 2]);
    }
};

} // namespace rhizome
