#include "WeightRAM.h"

// 1. Hàm nạp Weight từ AXI vào BRAM
void load_weights(
    hls::stream<weight_t> &in_weights,
    weight_t bram[16][1024], // Bỏ chiều [2], HLS sẽ tự biến nó thành Ping-Pong
    int num_weights_per_tile
) {
    // TỔNG số lượng phần tử cần nạp = số vector * 16 bank
    int total_elements = num_weights_per_tile * 16;
    
    for (int i = 0; i < total_elements; i++) {
        #pragma HLS PIPELINE II=1
        int bank_idx = i % 16;
        int addr = i / 16;
        bram[bank_idx][addr] = in_weights.read();
    }
}

// 2. Hàm đẩy Weight từ BRAM ra Engine (Có cơ chế Reuse)
void feed_weights(
    weight_t bram[16][1024],
    hls::stream<weight_mat_t> &out_weight_stream,
    int num_spatial_tiles, 
    int num_weights_per_tile
) {
    // VÒNG LẶP REUSE
    reuse_loop: for (int t = 0; t < num_spatial_tiles; t++) {
        
        // Vòng lặp xuất dữ liệu
        feed_loop: for (int w = 0; w < num_weights_per_tile; w++) {
            #pragma HLS PIPELINE II=1
            weight_mat_t temp_mat;
            
            for (int i = 0; i < 16; i++) {
                #pragma HLS UNROLL
                temp_mat.range(i * 8 + 7, i * 8) = bram[i][w];
            }
            out_weight_stream.write(temp_mat);
        }
    }
}

// 3. TOP MODULE
void weight_controller_top(
    hls::stream<weight_t> &in_weights,
    hls::stream<weight_mat_t> &out_weight_stream,
    hls::stream<int> &cfg_num_tiles, 
    hls::stream<int> &cfg_weight_size,
    hls::stream<int> &cfg_total_phases 
) {
    // Đọc cấu hình một lần ở đầu Layer
    int num_spatial_tiles = cfg_num_tiles.read();
    int num_weights_per_tile = cfg_weight_size.read();
    int total_phases = cfg_total_phases.read();

    // Vòng lặp chạy qua tất cả các Phase của toàn bộ Layer
    for (int phase = 0; phase < total_phases; phase++) {
        #pragma HLS DATAFLOW
        
        // Khai báo BRAM ngay TRONG vòng lặp DATAFLOW.
        // HLS sẽ tự động cấp phát nó thành cơ chế Ping-Pong Buffer (PIPO)
        // để overlap giữa việc load_weights (Phase i+1) và feed_weights (Phase i).
        weight_t local_bram[16][1024];
        #pragma HLS ARRAY_PARTITION variable=local_bram complete dim=1
        #pragma HLS BIND_STORAGE variable=local_bram type=ram_2p impl=bram

        // Nạp dữ liệu vào BRAM
        load_weights(in_weights, local_bram, num_weights_per_tile);
        
        // Đẩy dữ liệu ra Engine
        feed_weights(local_bram, out_weight_stream, num_spatial_tiles, num_weights_per_tile);
    }
}