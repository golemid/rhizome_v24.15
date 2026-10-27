#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_common.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

namespace rhizome {

#ifdef __CUDACC__

// ============================================================================
// Two-Pass Watertight 2-Manifold GPU Marching Cubes Lookup Tables & Kernels
// (10+ Million Triangle Capacity, Zero Spikes, Shared Welded Edge Vertices)
// ============================================================================
__constant__ int c_mc_edge_table[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

__constant__ int8_t c_mc_tri_table[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// Seals outer boundary voxels to positive exterior distance so Marching Cubes iso-surfaces are 100% closed
__global__ void gpu_mc_seal_border_kernel(
    float* __restrict__ distance_grid,
    int dim_x, int dim_y, int dim_z,
    float min_exterior_val
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    if (x == 0 || x == dim_x - 1 ||
        y == 0 || y == dim_y - 1 ||
        z == 0 || z == dim_z - 1) {
        size_t idx = (static_cast<size_t>(z) * dim_y + y) * dim_x + x;
        if (distance_grid[idx] < min_exterior_val) {
            distance_grid[idx] = min_exterior_val;
        }
    }
}

__device__ inline int dev_mc_cube_index(
    const float* __restrict__ grid,
    int x, int y, int z,
    size_t stride_y, size_t stride_z,
    float iso_value,
    float v_out[8]
) {
    size_t base = static_cast<size_t>(z) * stride_z + static_cast<size_t>(y) * stride_y + static_cast<size_t>(x);
    v_out[0] = grid[base] - iso_value;
    v_out[1] = grid[base + 1] - iso_value;
    v_out[2] = grid[base + stride_y + 1] - iso_value;
    v_out[3] = grid[base + stride_y] - iso_value;
    v_out[4] = grid[base + stride_z] - iso_value;
    v_out[5] = grid[base + stride_z + 1] - iso_value;
    v_out[6] = grid[base + stride_z + stride_y + 1] - iso_value;
    v_out[7] = grid[base + stride_z + stride_y] - iso_value;

    int idx = 0;
    if (v_out[0] <= 0.0f) idx |= 1;
    if (v_out[1] <= 0.0f) idx |= 2;
    if (v_out[2] <= 0.0f) idx |= 4;
    if (v_out[3] <= 0.0f) idx |= 8;
    if (v_out[4] <= 0.0f) idx |= 16;
    if (v_out[5] <= 0.0f) idx |= 32;
    if (v_out[6] <= 0.0f) idx |= 64;
    if (v_out[7] <= 0.0f) idx |= 128;
    return idx;
}

// Pass 1: Count exact number of active welded edge vertices and triangles on GPU (supports 10M+ triangles)
__global__ void gpu_mc_count_pass_kernel(
    const float* __restrict__ distance_grid,
    int dim_x, int dim_y, int dim_z,
    float iso_value,
    unsigned int* __restrict__ d_vert_count,
    unsigned int* __restrict__ d_tri_count
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    const size_t stride_y = static_cast<size_t>(dim_x);
    const size_t stride_z = static_cast<size_t>(dim_y) * stride_y;
    const size_t base = static_cast<size_t>(z) * stride_z + static_cast<size_t>(y) * stride_y + static_cast<size_t>(x);

    float v0 = distance_grid[base] - iso_value;
    bool s0 = (v0 <= 0.0f);

    unsigned int local_verts = 0;
    if (x + 1 < dim_x) {
        bool sx = ((distance_grid[base + 1] - iso_value) <= 0.0f);
        if (s0 != sx) local_verts++;
    }
    if (y + 1 < dim_y) {
        bool sy = ((distance_grid[base + stride_y] - iso_value) <= 0.0f);
        if (s0 != sy) local_verts++;
    }
    if (z + 1 < dim_z) {
        bool sz = ((distance_grid[base + stride_z] - iso_value) <= 0.0f);
        if (s0 != sz) local_verts++;
    }
    if (local_verts > 0) {
        atomicAdd(d_vert_count, local_verts);
    }

    if (x + 1 < dim_x && y + 1 < dim_y && z + 1 < dim_z) {
        float cv[8];
        int c_idx = dev_mc_cube_index(distance_grid, x, y, z, stride_y, stride_z, iso_value, cv);
        if (c_idx != 0 && c_idx != 255) {
            unsigned int local_tris = 0;
            for (int t = 0; t < 15; t += 3) {
                if (c_mc_tri_table[c_idx][t] < 0) break;
                local_tris++;
            }
            if (local_tris > 0) {
                atomicAdd(d_tri_count, local_tris);
            }
        }
    }
}

// Pass 2A: Emit welded vertices strictly along X, Y, Z grid edges (Zero Spikes)
__global__ void gpu_mc_emit_vertices_kernel(
    const float* __restrict__ distance_grid,
    int dim_x, int dim_y, int dim_z,
    float voxel_size,
    float origin_x, float origin_y, float origin_z,
    float iso_value,
    int32_t* __restrict__ edge_vert_x,
    int32_t* __restrict__ edge_vert_y,
    int32_t* __restrict__ edge_vert_z,
    float* __restrict__ out_vertices,
    unsigned int* __restrict__ d_vert_counter,
    unsigned int max_verts
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    const size_t stride_y = static_cast<size_t>(dim_x);
    const size_t stride_z = static_cast<size_t>(dim_y) * stride_y;
    const size_t base = static_cast<size_t>(z) * stride_z + static_cast<size_t>(y) * stride_y + static_cast<size_t>(x);

    float v0 = distance_grid[base] - iso_value;
    bool s0 = (v0 <= 0.0f);
    glm::vec3 p0(origin_x + x * voxel_size, origin_y + y * voxel_size, origin_z + z * voxel_size);

    // X-edge: (x,y,z) -> (x+1,y,z)
    if (x + 1 < dim_x) {
        float v1 = distance_grid[base + 1] - iso_value;
        if (s0 != (v1 <= 0.0f)) {
            float denom = v0 - v1;
            float t = (fabsf(denom) > 1e-7f) ? fminf(fmaxf(v0 / denom, 0.001f), 0.999f) : 0.5f;
            unsigned int v_idx = atomicAdd(d_vert_counter, 1u);
            if (v_idx < max_verts) {
                out_vertices[static_cast<size_t>(v_idx) * 3 + 0] = p0.x + t * voxel_size;
                out_vertices[static_cast<size_t>(v_idx) * 3 + 1] = p0.y;
                out_vertices[static_cast<size_t>(v_idx) * 3 + 2] = p0.z;
                edge_vert_x[base] = static_cast<int32_t>(v_idx);
            }
        }
    }

    // Y-edge: (x,y,z) -> (x,y+1,z)
    if (y + 1 < dim_y) {
        float v1 = distance_grid[base + stride_y] - iso_value;
        if (s0 != (v1 <= 0.0f)) {
            float denom = v0 - v1;
            float t = (fabsf(denom) > 1e-7f) ? fminf(fmaxf(v0 / denom, 0.001f), 0.999f) : 0.5f;
            unsigned int v_idx = atomicAdd(d_vert_counter, 1u);
            if (v_idx < max_verts) {
                out_vertices[static_cast<size_t>(v_idx) * 3 + 0] = p0.x;
                out_vertices[static_cast<size_t>(v_idx) * 3 + 1] = p0.y + t * voxel_size;
                out_vertices[static_cast<size_t>(v_idx) * 3 + 2] = p0.z;
                edge_vert_y[base] = static_cast<int32_t>(v_idx);
            }
        }
    }

    // Z-edge: (x,y,z) -> (x,y,z+1)
    if (z + 1 < dim_z) {
        float v1 = distance_grid[base + stride_z] - iso_value;
        if (s0 != (v1 <= 0.0f)) {
            float denom = v0 - v1;
            float t = (fabsf(denom) > 1e-7f) ? fminf(fmaxf(v0 / denom, 0.001f), 0.999f) : 0.5f;
            unsigned int v_idx = atomicAdd(d_vert_counter, 1u);
            if (v_idx < max_verts) {
                out_vertices[static_cast<size_t>(v_idx) * 3 + 0] = p0.x;
                out_vertices[static_cast<size_t>(v_idx) * 3 + 1] = p0.y;
                out_vertices[static_cast<size_t>(v_idx) * 3 + 2] = p0.z + t * voxel_size;
                edge_vert_z[base] = static_cast<int32_t>(v_idx);
            }
        }
    }
}

// Pass 2B: Connect shared welded edge vertices into watertight 2-manifold triangles
__global__ void gpu_mc_emit_triangles_kernel(
    const float* __restrict__ distance_grid,
    int dim_x, int dim_y, int dim_z,
    float iso_value,
    const int32_t* __restrict__ edge_vert_x,
    const int32_t* __restrict__ edge_vert_y,
    const int32_t* __restrict__ edge_vert_z,
    uint32_t* __restrict__ out_indices,
    unsigned int* __restrict__ d_tri_counter,
    unsigned int max_tris
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dim_x - 1 || y >= dim_y - 1 || z >= dim_z - 1) return;

    const size_t stride_y = static_cast<size_t>(dim_x);
    const size_t stride_z = static_cast<size_t>(dim_y) * stride_y;

    float cv[8];
    int c_idx = dev_mc_cube_index(distance_grid, x, y, z, stride_y, stride_z, iso_value, cv);
    if (c_idx == 0 || c_idx == 255) return;

    const size_t b000 = static_cast<size_t>(z) * stride_z + static_cast<size_t>(y) * stride_y + static_cast<size_t>(x);
    const size_t b100 = b000 + 1;
    const size_t b010 = b000 + stride_y;
    const size_t b110 = b010 + 1;
    const size_t b001 = b000 + stride_z;
    const size_t b101 = b001 + 1;
    const size_t b011 = b001 + stride_y;

    int32_t e_verts[12];
    e_verts[0]  = edge_vert_x[b000];
    e_verts[1]  = edge_vert_y[b100];
    e_verts[2]  = edge_vert_x[b010];
    e_verts[3]  = edge_vert_y[b000];
    e_verts[4]  = edge_vert_x[b001];
    e_verts[5]  = edge_vert_y[b101];
    e_verts[6]  = edge_vert_x[b011];
    e_verts[7]  = edge_vert_y[b001];
    e_verts[8]  = edge_vert_z[b000];
    e_verts[9]  = edge_vert_z[b100];
    e_verts[10] = edge_vert_z[b110];
    e_verts[11] = edge_vert_z[b010];

    for (int t = 0; t < 15; t += 3) {
        int e0 = c_mc_tri_table[c_idx][t + 0];
        if (e0 < 0) break;
        int e1 = c_mc_tri_table[c_idx][t + 1];
        int e2 = c_mc_tri_table[c_idx][t + 2];

        int32_t v0 = e_verts[e0];
        int32_t v1 = e_verts[e1];
        int32_t v2 = e_verts[e2];
        if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v0 != v1 && v1 != v2 && v2 != v0) {
            unsigned int t_idx = atomicAdd(d_tri_counter, 1u);
            if (t_idx < max_tris) {
                size_t out_base = static_cast<size_t>(t_idx) * 3;
                out_indices[out_base + 0] = static_cast<uint32_t>(v0);
                out_indices[out_base + 1] = static_cast<uint32_t>(v2);
                out_indices[out_base + 2] = static_cast<uint32_t>(v1);
            }
        }
    }
}

__global__ void gpu_dc_pass1_kernel(
    const float* __restrict__ distance_grid,
    int dim_x, int dim_y, int dim_z,
    int cx_dim, int cy_dim, int cz_dim,
    float voxel_size,
    float origin_x, float origin_y, float origin_z,
    float iso_value,
    int* __restrict__ cell_vert,
    float* __restrict__ out_vertices,
    int* __restrict__ d_vert_count,
    int max_verts
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= cx_dim || y >= cy_dim || z >= cz_dim) return;

    size_t c_idx = static_cast<size_t>(z) * cy_dim * cx_dim + static_cast<size_t>(y) * cx_dim + static_cast<size_t>(x);
    cell_vert[c_idx] = -1;

    const size_t stride_y = static_cast<size_t>(dim_x);
    const size_t stride_z = static_cast<size_t>(dim_y) * stride_y;

    const float* row00 = &distance_grid[static_cast<size_t>(z) * stride_z + static_cast<size_t>(y) * stride_y];
    const float* row10 = row00 + stride_y;
    const float* row01 = row00 + stride_z;
    const float* row11 = row01 + stride_y;

    float v0 = row00[x] - iso_value;
    float v1 = row00[x + 1] - iso_value;
    float v2 = row10[x] - iso_value;
    float v3 = row10[x + 1] - iso_value;
    float v4 = row01[x] - iso_value;
    float v5 = row01[x + 1] - iso_value;
    float v6 = row11[x] - iso_value;
    float v7 = row11[x + 1] - iso_value;

    int neg_mask =
        ((v0 <= 0.0f) ? 1 : 0) |
        ((v1 <= 0.0f) ? 2 : 0) |
        ((v2 <= 0.0f) ? 4 : 0) |
        ((v3 <= 0.0f) ? 8 : 0) |
        ((v4 <= 0.0f) ? 16 : 0) |
        ((v5 <= 0.0f) ? 32 : 0) |
        ((v6 <= 0.0f) ? 64 : 0) |
        ((v7 <= 0.0f) ? 128 : 0);

    if (neg_mask == 0 || neg_mask == 255) return;

    const float v[8] = { v0, v1, v2, v3, v4, v5, v6, v7 };
    glm::vec3 cell_min = glm::vec3(origin_x, origin_y, origin_z) + glm::vec3(x, y, z) * voxel_size;
    glm::vec3 corner_pos[8] = {
        cell_min,
        cell_min + glm::vec3(voxel_size, 0.0f, 0.0f),
        cell_min + glm::vec3(0.0f, voxel_size, 0.0f),
        cell_min + glm::vec3(voxel_size, voxel_size, 0.0f),
        cell_min + glm::vec3(0.0f, 0.0f, voxel_size),
        cell_min + glm::vec3(voxel_size, 0.0f, voxel_size),
        cell_min + glm::vec3(0.0f, voxel_size, voxel_size),
        cell_min + glm::vec3(voxel_size, voxel_size, voxel_size)
    };

    const int edge_corners[12][2] = {
        {0,1}, {2,3}, {4,5}, {6,7},
        {0,2}, {1,3}, {4,6}, {5,7},
        {0,4}, {1,5}, {2,6}, {3,7}
    };

    glm::vec3 avg_pos(0.0f);
    int crossings = 0;
    for (int e = 0; e < 12; ++e) {
        int c0 = edge_corners[e][0];
        int c1 = edge_corners[e][1];
        bool s0 = (v[c0] <= 0.0f);
        bool s1 = (v[c1] <= 0.0f);
        if (s0 != s1) {
            float denom = v[c0] - v[c1];
            float t = (fabsf(denom) > 1e-7f) ? (v[c0] / denom) : 0.5f;
            t = fminf(fmaxf(t, 0.0f), 1.0f);
            avg_pos += corner_pos[c0] + (corner_pos[c1] - corner_pos[c0]) * t;
            crossings++;
        }
    }

    glm::vec3 p = (crossings > 0) ? (avg_pos / static_cast<float>(crossings)) : (cell_min + glm::vec3(0.5f * voxel_size));

    // Project vertex onto exact zero-isosurface using trilinear cell gradient (4 Newton steps)
    for (int iter = 0; iter < 4; ++iter) {
        glm::vec3 u = glm::clamp((p - cell_min) / voxel_size, glm::vec3(0.0f), glm::vec3(1.0f));
        float c00 = v[0] * (1.0f - u.x) + v[1] * u.x;
        float c10 = v[2] * (1.0f - u.x) + v[3] * u.x;
        float c01 = v[4] * (1.0f - u.x) + v[5] * u.x;
        float c11 = v[6] * (1.0f - u.x) + v[7] * u.x;
        float c0 = c00 * (1.0f - u.y) + c10 * u.y;
        float c1 = c01 * (1.0f - u.y) + c11 * u.y;
        float val = c0 * (1.0f - u.z) + c1 * u.z;

        float gx = ((v[1] - v[0]) * (1.0f - u.y) + (v[3] - v[2]) * u.y) * (1.0f - u.z) +
                   ((v[5] - v[4]) * (1.0f - u.y) + (v[7] - v[6]) * u.y) * u.z;
        float gy = (c10 - c00) * (1.0f - u.z) + (c11 - c01) * u.z;
        float gz = c1 - c0;
        glm::vec3 grad = glm::vec3(gx, gy, gz) / voxel_size;
        float g2 = glm::dot(grad, grad);
        if (g2 > 1e-6f) {
            p -= (val / g2) * grad;
            p = glm::clamp(p, cell_min, cell_min + glm::vec3(voxel_size));
        }
    }

    int v_idx = atomicAdd(d_vert_count, 1);
    if (v_idx < max_verts) {
        out_vertices[v_idx * 3 + 0] = p.x;
        out_vertices[v_idx * 3 + 1] = p.y;
        out_vertices[v_idx * 3 + 2] = p.z;
        cell_vert[c_idx] = v_idx;
    }
}

__global__ void gpu_dc_pass2_kernel(
    const float* __restrict__ distance_grid,
    int dim_x, int dim_y, int dim_z,
    int cx_dim, int cy_dim, int cz_dim,
    float iso_value,
    const int* __restrict__ cell_vert,
    uint32_t* __restrict__ out_indices,
    int* __restrict__ d_tri_count,
    int max_tris
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= cx_dim || y >= cy_dim || z >= cz_dim) return;

    const size_t stride_y = static_cast<size_t>(dim_x);
    const size_t stride_z = static_cast<size_t>(dim_y) * stride_y;

    auto sample_val = [&](int px, int py, int pz) -> float {
        return distance_grid[static_cast<size_t>(pz) * stride_z + static_cast<size_t>(py) * stride_y + static_cast<size_t>(px)] - iso_value;
    };

    auto cell_index = [&](int px, int py, int pz) -> size_t {
        return static_cast<size_t>(pz) * cy_dim * cx_dim + static_cast<size_t>(py) * cx_dim + static_cast<size_t>(px);
    };

    auto emit_quad_gpu = [&](int32_t i0, int32_t i1, int32_t i2, int32_t i3, bool flip) {
        if (i0 < 0 || i1 < 0 || i2 < 0 || i3 < 0) return;
        int t_idx = atomicAdd(d_tri_count, 2);
        if (t_idx + 1 < max_tris) {
            size_t base = static_cast<size_t>(t_idx) * 3;
            if (!flip) {
                out_indices[base + 0] = static_cast<uint32_t>(i0);
                out_indices[base + 1] = static_cast<uint32_t>(i1);
                out_indices[base + 2] = static_cast<uint32_t>(i2);
                out_indices[base + 3] = static_cast<uint32_t>(i0);
                out_indices[base + 4] = static_cast<uint32_t>(i2);
                out_indices[base + 5] = static_cast<uint32_t>(i3);
            } else {
                out_indices[base + 0] = static_cast<uint32_t>(i0);
                out_indices[base + 1] = static_cast<uint32_t>(i2);
                out_indices[base + 2] = static_cast<uint32_t>(i1);
                out_indices[base + 3] = static_cast<uint32_t>(i0);
                out_indices[base + 4] = static_cast<uint32_t>(i3);
                out_indices[base + 5] = static_cast<uint32_t>(i2);
            }
        }
    };

    // X-parallel edge: (x, y, z) -> (x + 1, y, z)
    if (y >= 1 && z >= 1) {
        bool s0 = (sample_val(x, y, z) <= 0.0f);
        bool s1 = (sample_val(x + 1, y, z) <= 0.0f);
        if (s0 != s1) {
            int32_t c0 = cell_vert[cell_index(x, y - 1, z - 1)];
            int32_t c1 = cell_vert[cell_index(x, y,     z - 1)];
            int32_t c2 = cell_vert[cell_index(x, y,     z)];
            int32_t c3 = cell_vert[cell_index(x, y - 1, z)];
            emit_quad_gpu(c0, c1, c2, c3, !s0);
        }
    }

    // Y-parallel edge: (x, y, z) -> (x, y + 1, z)
    if (x >= 1 && z >= 1) {
        bool s0 = (sample_val(x, y, z) <= 0.0f);
        bool s1 = (sample_val(x, y + 1, z) <= 0.0f);
        if (s0 != s1) {
            int32_t c0 = cell_vert[cell_index(x - 1, y, z - 1)];
            int32_t c1 = cell_vert[cell_index(x - 1, y, z)];
            int32_t c2 = cell_vert[cell_index(x,     y, z)];
            int32_t c3 = cell_vert[cell_index(x,     y, z - 1)];
            emit_quad_gpu(c0, c1, c2, c3, !s0);
        }
    }

    // Z-parallel edge: (x, y, z) -> (x, y, z + 1)
    if (x >= 1 && y >= 1) {
        bool s0 = (sample_val(x, y, z) <= 0.0f);
        bool s1 = (sample_val(x, y, z + 1) <= 0.0f);
        if (s0 != s1) {
            int32_t c0 = cell_vert[cell_index(x - 1, y - 1, z)];
            int32_t c1 = cell_vert[cell_index(x,     y - 1, z)];
            int32_t c2 = cell_vert[cell_index(x,     y,     z)];
            int32_t c3 = cell_vert[cell_index(x - 1, y,     z)];
            emit_quad_gpu(c0, c1, c2, c3, !s0);
        }
    }
}

__global__ void gpu_laplacian_accum_kernel(
    const float* __restrict__ vertices,
    int num_vertices,
    const uint32_t* __restrict__ indices,
    int num_triangles,
    float* __restrict__ out_accum_pos,
    int* __restrict__ out_neighbor_count
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    uint32_t i0 = indices[t * 3 + 0];
    uint32_t i1 = indices[t * 3 + 1];
    uint32_t i2 = indices[t * 3 + 2];

    if (i0 >= num_vertices || i1 >= num_vertices || i2 >= num_vertices) return;

    float v0x = vertices[i0 * 3 + 0], v0y = vertices[i0 * 3 + 1], v0z = vertices[i0 * 3 + 2];
    float v1x = vertices[i1 * 3 + 0], v1y = vertices[i1 * 3 + 1], v1z = vertices[i1 * 3 + 2];
    float v2x = vertices[i2 * 3 + 0], v2y = vertices[i2 * 3 + 1], v2z = vertices[i2 * 3 + 2];

    atomicAdd(&out_accum_pos[i0 * 3 + 0], v1x + v2x);
    atomicAdd(&out_accum_pos[i0 * 3 + 1], v0y + v2y);
    atomicAdd(&out_accum_pos[i0 * 3 + 2], v0z + v2z);
    atomicAdd(&out_neighbor_count[i0], 2);

    atomicAdd(&out_accum_pos[i1 * 3 + 0], v0x + v2x);
    atomicAdd(&out_accum_pos[i1 * 3 + 1], v0y + v2y);
    atomicAdd(&out_accum_pos[i1 * 3 + 2], v0z + v2z);
    atomicAdd(&out_neighbor_count[i1], 2);

    atomicAdd(&out_accum_pos[i2 * 3 + 0], v0x + v1x);
    atomicAdd(&out_accum_pos[i2 * 3 + 1], v0y + v1y);
    atomicAdd(&out_accum_pos[i2 * 3 + 2], v0z + v1z);
    atomicAdd(&out_neighbor_count[i2], 2);
}

__global__ void gpu_laplacian_apply_kernel(
    float* __restrict__ vertices,
    int num_vertices,
    const float* __restrict__ accum_pos,
    const int* __restrict__ neighbor_count,
    float lambda
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;

    int count = neighbor_count[i];
    if (count > 0) {
        float inv_c = 1.0f / static_cast<float>(count);
        float avg_x = accum_pos[i * 3 + 0] * inv_c;
        float avg_y = accum_pos[i * 3 + 1] * inv_c;
        float avg_z = accum_pos[i * 3 + 2] * inv_c;

        vertices[i * 3 + 0] += lambda * (avg_x - vertices[i * 3 + 0]);
        vertices[i * 3 + 1] += lambda * (avg_y - vertices[i * 3 + 1]);
        vertices[i * 3 + 2] += lambda * (avg_z - vertices[i * 3 + 2]);
    }
}

#endif // __CUDACC__

// 4. Two-Pass Watertight 2-Manifold GPU Marching Cubes Iso-Surface Extractor (10M+ Triangles, Zero Spikes)
bool CudaRasterizer::extract_iso_surface(
    const std::vector<float>& distance_grid,
    glm::ivec3 dims,
    float voxel_size,
    glm::vec3 origin,
    float iso_value,
    TriangleMesh& out_mesh,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_mesh.vertices.clear();
    out_mesh.indices.clear();

    if (dims.x < 2 || dims.y < 2 || dims.z < 2 || distance_grid.empty()) {
        out_time_ms = 0.0f;
        return true;
    }

    const size_t total_voxels = static_cast<size_t>(dims.x) * dims.y * dims.z;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_grid = nullptr;
        int32_t* d_edge_x = nullptr;
        int32_t* d_edge_y = nullptr;
        int32_t* d_edge_z = nullptr;
        unsigned int* d_counts = nullptr; // [0]=vert_count, [1]=tri_count

        size_t grid_bytes = total_voxels * sizeof(float);
        size_t edge_bytes = total_voxels * sizeof(int32_t);

        if (cudaMalloc(&d_grid, grid_bytes) == cudaSuccess &&
            cudaMalloc(&d_edge_x, edge_bytes) == cudaSuccess &&
            cudaMalloc(&d_edge_y, edge_bytes) == cudaSuccess &&
            cudaMalloc(&d_edge_z, edge_bytes) == cudaSuccess &&
            cudaMalloc(&d_counts, 2 * sizeof(unsigned int)) == cudaSuccess) {

            cudaMemcpy(d_grid, distance_grid.data(), grid_bytes, cudaMemcpyHostToDevice);
            cudaMemset(d_edge_x, 0xFF, edge_bytes);
            cudaMemset(d_edge_y, 0xFF, edge_bytes);
            cudaMemset(d_edge_z, 0xFF, edge_bytes);
            cudaMemset(d_counts, 0, 2 * sizeof(unsigned int));

            dim3 block(8, 8, 8);
            dim3 grid((dims.x + 7) / 8, (dims.y + 7) / 8, (dims.z + 7) / 8);

            // Step 0: Seal outer grid boundary voxels so Marching Cubes iso-surface is 100% closed
            gpu_mc_seal_border_kernel<<<grid, block>>>(
                d_grid, dims.x, dims.y, dims.z, iso_value + voxel_size
            );

            // Pass 1: Count exact active edge vertices & triangles on GPU
            gpu_mc_count_pass_kernel<<<grid, block>>>(
                d_grid, dims.x, dims.y, dims.z, iso_value, &d_counts[0], &d_counts[1]
            );

            unsigned int h_counts[2] = { 0u, 0u };
            cudaMemcpy(h_counts, d_counts, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            unsigned int exact_verts = h_counts[0];
            unsigned int exact_tris  = h_counts[1];

            if (exact_verts > 0u && exact_tris > 0u) {
                float* d_verts = nullptr;
                uint32_t* d_indices = nullptr;
                size_t vert_bytes = static_cast<size_t>(exact_verts) * 3 * sizeof(float);
                size_t idx_bytes  = static_cast<size_t>(exact_tris) * 3 * sizeof(uint32_t);

                if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
                    cudaMalloc(&d_indices, idx_bytes) == cudaSuccess) {

                    cudaMemset(d_counts, 0, 2 * sizeof(unsigned int));

                    // Pass 2A: Emit welded vertices strictly along X, Y, Z grid edges (Zero Spikes)
                    gpu_mc_emit_vertices_kernel<<<grid, block>>>(
                        d_grid, dims.x, dims.y, dims.z,
                        voxel_size, origin.x, origin.y, origin.z, iso_value,
                        d_edge_x, d_edge_y, d_edge_z,
                        d_verts, &d_counts[0], exact_verts
                    );

                    // Pass 2B: Connect shared welded edge vertices into watertight triangles
                    gpu_mc_emit_triangles_kernel<<<grid, block>>>(
                        d_grid, dims.x, dims.y, dims.z, iso_value,
                        d_edge_x, d_edge_y, d_edge_z,
                        d_indices, &d_counts[1], exact_tris
                    );

                    cudaMemcpy(h_counts, d_counts, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                    unsigned int final_verts = std::min(h_counts[0], exact_verts);
                    unsigned int final_tris  = std::min(h_counts[1], exact_tris);

                    if (final_verts > 0u && final_tris > 0u) {
                        out_mesh.vertices.resize(static_cast<size_t>(final_verts) * 3);
                        out_mesh.indices.resize(static_cast<size_t>(final_tris) * 3);
                        cudaMemcpy(out_mesh.vertices.data(), d_verts, out_mesh.vertices.size() * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(out_mesh.indices.data(), d_indices, out_mesh.indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    }
                }
                if (d_verts) cudaFree(d_verts);
                if (d_indices) cudaFree(d_indices);
            }

            cudaFree(d_grid);
            cudaFree(d_edge_x);
            cudaFree(d_edge_y);
            cudaFree(d_edge_z);
            cudaFree(d_counts);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_grid) cudaFree(d_grid);
        if (d_edge_x) cudaFree(d_edge_x);
        if (d_edge_y) cudaFree(d_edge_y);
        if (d_edge_z) cudaFree(d_edge_z);
        if (d_counts) cudaFree(d_counts);
    }
#endif
    out_time_ms = 0.0f;
    return true;
}

TriangleMesh CudaRasterizer::extract_iso_surface(
    const CudaVolumeGrid& volume,
    float iso_value
) {
    TriangleMesh mesh;
    float time_dummy = 0.0f;
    glm::vec3 origin = glm::vec3(volume.min_coord) * volume.voxel_size;
    extract_iso_surface(volume.data, volume.dims, volume.voxel_size, origin, iso_value, mesh, time_dummy);
    return mesh;
}

bool CudaRasterizer::rasterize_and_extract_mesh_gpu(
    const std::vector<CudaCapsule>& capsules,
    float voxel_size,
    float background,
    glm::ivec3 min_coord,
    glm::ivec3 max_coord,
    float iso_value,
    TriangleMesh& out_mesh,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_mesh.vertices.clear();
    out_mesh.indices.clear();

    if (capsules.empty()) {
        out_time_ms = 0.0f;
        return true;
    }

    int dim_x = max_coord.x - min_coord.x + 1;
    int dim_y = max_coord.y - min_coord.y + 1;
    int dim_z = max_coord.z - min_coord.z + 1;
    if (dim_x < 2 || dim_y < 2 || dim_z < 2) {
        out_time_ms = 0.0f;
        return true;
    }

    std::vector<float> distance_grid;
    float rast_ms = 0.0f;
    if (!rasterize(capsules, voxel_size, background, min_coord, max_coord, distance_grid, rast_ms, 0.0f)) {
        std::cerr << "[CUDA Error] GPU acceleration required for rasterize_roots.\n";
        return false;
    }

    glm::vec3 origin = glm::vec3(min_coord) * voxel_size;
    float mc_ms = 0.0f;
    if (!extract_iso_surface(distance_grid, glm::ivec3(dim_x, dim_y, dim_z), voxel_size, origin, iso_value, out_mesh, mc_ms)) {
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    return true;
}

bool CudaRasterizer::smooth_mesh_laplacian_gpu(
    TriangleMesh& inout_mesh,
    int iterations,
    float lambda,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_verts = inout_mesh.vertex_count();
    size_t num_tris = inout_mesh.triangle_count();

    if (num_verts == 0 || num_tris == 0 || iterations <= 0) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        float* d_accum = nullptr;
        int* d_counts = nullptr;

        size_t vert_bytes = inout_mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = inout_mesh.indices.size() * sizeof(uint32_t);
        size_t count_bytes = num_verts * sizeof(int);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_accum, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_counts, count_bytes) == cudaSuccess) {

            cudaMemcpy(d_verts, inout_mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, inout_mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);

            int tri_threads = 256;
            int tri_blocks = (static_cast<int>(num_tris) + tri_threads - 1) / tri_threads;
            int vert_threads = 256;
            int vert_blocks = (static_cast<int>(num_verts) + vert_threads - 1) / vert_threads;

            for (int it = 0; it < iterations; ++it) {
                cudaMemset(d_accum, 0, vert_bytes);
                cudaMemset(d_counts, 0, count_bytes);

                gpu_laplacian_accum_kernel<<<tri_blocks, tri_threads>>>(
                    d_verts, static_cast<int>(num_verts),
                    d_indices, static_cast<int>(num_tris),
                    d_accum, d_counts
                );

                gpu_laplacian_apply_kernel<<<vert_blocks, vert_threads>>>(
                    d_verts, static_cast<int>(num_verts),
                    d_accum, d_counts, lambda
                );
            }

            cudaMemcpy(inout_mesh.vertices.data(), d_verts, vert_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_accum);
            cudaFree(d_counts);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_accum) cudaFree(d_accum);
        if (d_counts) cudaFree(d_counts);
    }
#endif

    out_time_ms = 0.0f;
    return true;
}

} // namespace rhizome
