#include "Scheduler.h"

void scheduler_top(
    const LayerDescriptor& descriptor,
    bool start,
    hls::stream<ap_uint<2>>& engine_select_sys,
    hls::stream<ap_uint<2>>& engine_select_wino,
    hls::stream<ap_uint<2>>& engine_select_lb,
    hls::stream<ap_uint<2>>& engine_select_ddemux,
    hls::stream<ap_uint<2>>& engine_select_wdemux,
    hls::stream<ap_uint<2>>& engine_select_fuse
) {
    #pragma HLS INTERFACE s_axilite port=descriptor
    #pragma HLS INTERFACE s_axilite port=start
    #pragma HLS INTERFACE s_axilite port=return

    if (start) {
        ap_uint<2> mode = 0; 
        if (descriptor.kernel_size == 3 && descriptor.stride == 1) {
            mode = 0; // Winograd 3x3 s1
        } else if (descriptor.kernel_size == 3 && descriptor.stride == 2) {
            mode = 1; // Systolic 3x3 s2
        } else if (descriptor.kernel_size == 1 && descriptor.stride == 2) {
            mode = 2; // Systolic 1x1 s2
        } else {
            mode = descriptor.preferred_engine; 
        }
        
        engine_select_sys.write(mode);
        engine_select_wino.write(mode);
        engine_select_lb.write(mode);
        engine_select_ddemux.write(mode);
        engine_select_wdemux.write(mode);
        engine_select_fuse.write(mode);
    }
}

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
) {
    int W = descriptor.W;
    int H = descriptor.H;
    int Cin = descriptor.Cin;
    int Cout = descriptor.Cout;
    
    int tiles_X = (W - 4) / 2 + 1;
    int tiles_Y = (H - 4) / 2 + 1;
    int total_tiles_per_channel = tiles_X * tiles_Y;
    
    int total_layer_tiles = total_tiles_per_channel * Cin * Cout;
    
    ap_uint<2> mode = 0;
    if (descriptor.kernel_size == 3 && descriptor.stride == 1) {
        mode = 0; // Winograd 3x3 s1
    } else if (descriptor.kernel_size == 3 && descriptor.stride == 2) {
        mode = 1; // Systolic 3x3 s2
    } else if (descriptor.kernel_size == 1 && descriptor.stride == 2) {
        mode = 2; // Systolic 1x1 s2
    } else {
        mode = descriptor.preferred_engine;
    }

    // --- BIẾN CẤU HÌNH ĐỘNG THEO MODE ---
    int ddemux_tiles_val = 0;
    int wdemux_weights_val = 0;
    int wino_tiles_val = 0;
    
    int wr_tiles_val = 0;
    int wr_weights_val = 0;
    int wr_phases_val = 1;

    if (mode == 0) {
        // --- CHẾ ĐỘ WINOGRAD (Theo đúng tb_subsystems) ---
        ddemux_tiles_val = total_tiles_per_channel * Cin;         // DataRouter nhận: Tiles Không gian x Cin
        wdemux_weights_val = total_tiles_per_channel * Cin * Cout;// DataRouter phân luồng: Tiles Không gian x Cin x Cout
        wino_tiles_val = total_tiles_per_channel;                 // Vòng lặp Engine lặp theo Tiles Không gian
        
        wr_tiles_val = total_tiles_per_channel;                   // Cấu hình WeightRAM 
        wr_weights_val = Cin * Cout;                              // Số vectors trọng số cho mỗi Tile
        wr_phases_val = 1;
    } else {
        // --- CHẾ ĐỘ SYSTOLIC ---
        ddemux_tiles_val = total_layer_tiles; // Systolic yêu cầu toàn bộ 
        
        if (mode == 1) {
            wdemux_weights_val = 9 * Cin * Cout; // Systolic 3x3
        } else if (mode == 2) {
            wdemux_weights_val = 1 * Cin * Cout; // Systolic 1x1
        }
        
        wr_tiles_val = 1; 
        wr_weights_val = wdemux_weights_val;
        wr_phases_val = 1;
        wino_tiles_val = total_tiles_per_channel; // Giữ an toàn tránh rác
    }

    int expected_packets_per_ch = (mode == 0) ? ((total_tiles_per_channel + 3) / 4) : total_tiles_per_channel;
    int total_expected_packets = expected_packets_per_ch * Cin * Cout;
    
    // Bắn dữ liệu ra streams
    stream_lb_W.write(W);
    stream_lb_H.write(H);
    stream_lb_cin.write(Cin);

    // Bắn cho Weight RAM
    stream_wr_tiles.write(wr_tiles_val);
    stream_wr_weights.write(wr_weights_val);
    stream_wr_phases.write(wr_phases_val);

    stream_cout.write(Cout);
    
    stream_ddemux_tiles.write(ddemux_tiles_val);
    stream_wdemux_weights.write(wdemux_weights_val);
    stream_router2_tiles.write(total_layer_tiles);
    
    stream_sys_cin.write(Cin);
    stream_sys_cout.write(Cout);
    stream_sys_tiles_per_ch.write(total_tiles_per_channel);
    
    // Bắn cho Winograd
    stream_wino_tiles.write(wino_tiles_val);
    stream_wino_cin.write(Cin);
    stream_wino_cout.write(Cout);
    
    stream_acc_packets.write(total_expected_packets);
    stream_acc_cin.write(Cin);
    stream_fuse_cout.write(Cout);
}