#ifndef WEIGHTRAM_H
#define WEIGHTRAM_H

#include "global.h"

void load_weights(
    hls::stream<weight_t> &in_weights,
    weight_t bram[16][1024], // Bỏ chiều [2], HLS sẽ tự biến nó thành Ping-Pong
    int num_weights_per_tile
);

void feed_weights(
    weight_t bram[16][1024],
    hls::stream<weight_mat_t> &out_weight_stream,
    int num_spatial_tiles, 
    int num_weights_per_tile
);

void weight_controller_top(
    hls::stream<weight_t> &in_weights,
    hls::stream<weight_mat_t> &out_weight_stream,
    hls::stream<int> &cfg_num_tiles, 
    hls::stream<int> &cfg_weight_size,
    hls::stream<int> &cfg_total_phases 
);

#endif