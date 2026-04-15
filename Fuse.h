#ifndef FUSE_H
#define FUSE_H

#include "global.h"

void fuse_post_conv(
    hls::stream<fuse_vec_in_t>& conv_in,
    hls::stream<fuse_vec_out_t>& fuse_out,
    const ap_int<32> bias[MAX_CHANNELS],
    const ap_int<32> requant_mul[MAX_CHANNELS],
    const ap_int<8> requant_shift[MAX_CHANNELS],
    hls::stream<int>& channels_stream,
    bool& accel_done
);

void accumulator_top(
    hls::stream<fuse_vec_in_t>& in_mux_data,
    hls::stream<fuse_vec_in_t>& out_fuse_data,
    hls::stream<int>& total_tiles_stream,
    hls::stream<int>& cin_iters_stream 
);

void compute_to_fuse_serializer(
    hls::stream<psum_block_t>& systolic_in,
    hls::stream<Tile2x2>& winograd_in,
    hls::stream<fuse_vec_in_t>& fuse_out,
    hls::stream<ap_uint<2>>& mode_stream,
    hls::stream<int>& total_elements_stream
);

void fuse_post_conv(
    hls::stream<fuse_vec_in_t>& conv_in,
    hls::stream<fuse_vec_in_t>& residual_in, // Stream dữ liệu nhánh identity
    hls::stream<fuse_vec_out_t>& fuse_out,
    const ap_int<32> bias[MAX_CHANNELS],
    const ap_int<32> requant_mul[MAX_CHANNELS],
    const ap_int<8> requant_shift[MAX_CHANNELS],
    hls::stream<int>& channels_stream,
    hls::stream<fuse_config_t>& config_stream, // Cấu hình linh hoạt
    bool& accel_done
);

#endif