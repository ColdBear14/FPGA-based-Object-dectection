#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "global.h"

void scheduler_top(
    const LayerDescriptor& descriptor,
    bool start,
    hls::stream<ap_uint<2>>& engine_select_sys,
    hls::stream<ap_uint<2>>& engine_select_wino,
    hls::stream<ap_uint<2>>& engine_select_lb,
    hls::stream<ap_uint<2>>& engine_select_ddemux,
    hls::stream<ap_uint<2>>& engine_select_wdemux,
    hls::stream<ap_uint<2>>& engine_select_fuse
);

void config_decoder(
    LayerDescriptor descriptor,
    
    hls::stream<int>& stream_lb_W,
    hls::stream<int>& stream_lb_H,
    hls::stream<int>& stream_lb_cin,
    
    // THÊM: Các streams cấu hình cho Weight RAM bị thiếu
    hls::stream<int>& stream_wr_tiles,
    hls::stream<int>& stream_wr_weights,
    hls::stream<int>& stream_wr_phases,

    hls::stream<int>& stream_cout,
    
    hls::stream<int>& stream_ddemux_tiles,
    hls::stream<int>& stream_wdemux_weights,
    hls::stream<int>& stream_router2_tiles,
    
    hls::stream<int>& stream_sys_cin,
    hls::stream<int>& stream_sys_cout,
    hls::stream<int>& stream_sys_tiles_per_ch,
    
    hls::stream<int>& stream_wino_tiles,
    // THÊM: Các streams cấu hình kênh cho Winograd
    hls::stream<int>& stream_wino_cin,
    hls::stream<int>& stream_wino_cout,
    
    hls::stream<int>& stream_acc_packets,
    hls::stream<int>& stream_acc_cin,
    hls::stream<int>& stream_fuse_cout
);

#endif // SCHEDULER_H