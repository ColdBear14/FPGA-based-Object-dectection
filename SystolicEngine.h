#ifndef SYSTOLICENGINE_H
#define SYSTOLICENGINE_H

#include "global.h"

using namespace hls;

void apply_input_skew(stream<Tile16x16> &in, pixel_t out_skewed[ARRAY_SIZE]);

void apply_output_deskew(psum_t paa_outputs[ARRAY_SIZE], stream<psum_block_t> &out);

void apply_post_array_accumulator(psum_t psum_row_16[ARRAY_SIZE], psum_t paa_outputs[ARRAY_SIZE], ap_uint<3> ctrl);


void systolic_array_compute_simulation(
    pixel_t skewed_pixels[ARRAY_SIZE],
    weight_t pe_weight_buffer[ARRAY_SIZE][ARRAY_SIZE][KERNEL_SIZE],
    ap_uint<4> pe_rd_ptr[ARRAY_SIZE][ARRAY_SIZE],
    pixel_t pixel_stage[ARRAY_SIZE][ARRAY_SIZE+1], 
    psum_t psum_stage[ARRAY_SIZE+1][ARRAY_SIZE],
    ap_uint<3> ctrl
);

void systolic_engine(
    hls::stream<Tile16x16> &pixels_in, // Cập nhật stream
    hls::stream<weight_mat_t> &weights_in, 
    ap_uint<3> ctrl, 
    hls::stream<psum_block_t> &psums_out,
    hls::stream<ap_uint<2>> &mode_stream,
    hls::stream<int> &stream_cin,          
    hls::stream<int> &stream_cout,         
    hls::stream<int> &stream_tiles_per_ch  
);

#endif