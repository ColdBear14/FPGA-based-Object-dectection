#ifndef GLOBAL_H
#define GLOBAL_H

#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>

using namespace hls;
using namespace std;

// ==========================================================
// 1. SYSTEM CONFIGURATION
// ==========================================================
#define MAX_CHANNELS 1024
#define MAX_WIDTH    512
#define VECTOR_WIDTH 32   // 4 banks * 8-bit
#define MAX_TILES 2048
#define MAX_CIN 32
#define MAX_COUT 32

// ==========================================================
// 2. BASIC DATA TYPES
// ==========================================================
typedef ap_int<8>  pixel_t;     // Pixel / activation 8-bit
typedef ap_int<8>  weight_t;    // Weight 8-bit
typedef ap_int<32> data_t;      // Intermediate / accumulator data
typedef ap_int<32> psum_t;      // Partial sum for MAC blocks

// ==========================================================
// 3. AXI-STREAM INTERFACES
// ==========================================================
struct axis_in_t {
    ap_int<32> data;
    ap_uint<1> last;
};

struct axis_out_t {
    ap_int<8> data;
    ap_uint<1> last;
};

// ==========================================================
// 4. SYSTOLIC ARRAY & WEIGHT RAM
// ==========================================================
const int ARRAY_SIZE  = 16;  // 16x16 PE array
const int KERNEL_SIZE = 9;
const int NUM_BANKS   = 4;
const int BANK_DEPTH  = 1024;

typedef ap_uint<128> weight_mat_t;

struct psum_block_t {
    psum_t data[ARRAY_SIZE];
};

struct Column16 {
    pixel_t data[16];
};

// ==========================================================
// 5. WINOGRAD TRANSFORM STRUCTURES
// ==========================================================
struct Tile16x16 {
    data_t data[16][16];
};

struct Tile4x4 {
    data_t data[4][4];
};

struct Tile3x3 {
    data_t data[3][3];
};

struct Tile2x2 {
    data_t data[2][2];
};

// ==========================================================
// 6. FUSE MODULE
// ==========================================================
#define FUSE_PARALLEL_SIZE 16

struct fuse_vec_in_t {
    ap_int<32> data[FUSE_PARALLEL_SIZE];
    bool last;
};

struct fuse_vec_out_t {
    ap_int<8> data[FUSE_PARALLEL_SIZE];
    bool last;
};

// ==========================================================
// 7. FSM STATES
// ==========================================================
enum class SchedState {
    IDLE = 0,
    PREFETCH,
    DRAIN_AND_DONE
};

// ==========================================================
// 8. LAYER DESCRIPTOR
// ==========================================================
struct LayerDescriptor {
    ap_uint<2>  type;             // 0: Conv2D
    ap_uint<16> W;                // Feature map width
    ap_uint<16> H;                // Feature map height
    ap_uint<16> Cin;              // Input channels
    ap_uint<16> Cout;             // Output channels
    ap_uint<2>  kernel_size;      // 1: 1x1, 3: 3x3
    ap_uint<2>  stride;           // Stride (1 or 2)
    ap_uint<2>  pad;
    ap_uint<2>  dilation;
    ap_uint<32> weight_ptr;       // Pointer to weight memory


    ap_uint<1>  preferred_engine; // 0: Winograd, 1: Systolic
    ap_uint<8>  tile_policy;      // Tile partitioning policy
};

struct fuse_config_t {
    bool has_residual;   // Có thực hiện cộng Skip Connection không
    bool has_maxpool;    // Có thực hiện MaxPool 2x2 không
    int cat_offset;      // Vị trí bắt đầu ghi trong bộ nhớ (hỗ trợ Cat/Slice)
    int channel_limit;   // Giới hạn kênh (hỗ trợ Slice)
};

#endif // GLOBAL_H