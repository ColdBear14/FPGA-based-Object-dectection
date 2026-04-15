#include "DataRouter.h"

void weight_demux(
    hls::stream<weight_mat_t> &in_weight_stream, 
    hls::stream<ap_uint<2>>& mode_stream,                    
    hls::stream<weight_mat_t> &out_systolic_w,   
    hls::stream<Tile4x4> &out_winograd_w,        
    hls::stream<int>& num_weight_vectors_stream                
) {
    #pragma HLS INLINE off

    // Đọc thông số cấu hình 1 lần
    int num_weight_vectors = num_weight_vectors_stream.read();
    ap_uint<2> sel_val = mode_stream.read();

    if (sel_val == 0) { // Chế độ Winograd Engine
        demux_w_wino_loop: for (int i = 0; i < num_weight_vectors; i++) {
            #pragma HLS PIPELINE II=1
            weight_mat_t w_vec = in_weight_stream.read();
            Tile4x4 temp_w_tile;
            
            // Unpack vector 128-bit thành 16 phần tử riêng biệt cho ma trận 4x4
            unpack_row: for (int r = 0; r < 4; r++) {
                #pragma HLS UNROLL
                unpack_col: for (int c = 0; c < 4; c++) {
                    #pragma HLS UNROLL
                    int idx = r * 4 + c;
                    
                    temp_w_tile.data[r][c] = w_vec.range(idx * 8 + 7, idx * 8); 
                }
            }
            out_winograd_w.write(temp_w_tile);
        }
    }
    else { // Chế độ Systolic Array
        // Chuyển tiếp (forward) trực tiếp luồng weight 128-bit qua Stream.
        // Vector chứa 16 phần tử này sẽ được cấp thẳng vào mảng Systolic 16x16.
        demux_w_sys_loop: for (int i = 0; i < num_weight_vectors; i++) {
            #pragma HLS PIPELINE II=1
            out_systolic_w.write(in_weight_stream.read());
        }
    }
}

void data_demux(
    hls::stream<Tile4x4> &in_stream,     
    hls::stream<ap_uint<2>>& mode_stream,            
    hls::stream<Tile4x4> &out_systolic,  
    hls::stream<Tile4x4> &out_winograd,  
    hls::stream<int>& num_tiles_stream
) {
    #pragma HLS INLINE off
    
    // Đọc tĩnh 1 lần
    int num_tiles = num_tiles_stream.read();
    ap_uint<2> sel_val = mode_stream.read();
    
    if (sel_val == 0) {
        demux_wino_loop: for (int i = 0; i < num_tiles; i++) {
            #pragma HLS PIPELINE II=1
            out_winograd.write(in_stream.read());
        }
    }
    else if(sel_val == 1) {
        demux_sys_1_loop: for (int i = 0; i < num_tiles; i++) {
            #pragma HLS PIPELINE II=1
            out_systolic.write(in_stream.read());
        }
    }
    else {
        demux_sys_2_loop: for (int i = 0; i < num_tiles; i++) {
            #pragma HLS PIPELINE II=1
            out_systolic.write(in_stream.read());
        }
    }

}

