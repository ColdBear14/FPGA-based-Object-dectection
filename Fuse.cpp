#include "Fuse.h"

void fuse_post_conv(
    hls::stream<fuse_vec_in_t>& conv_in,
    hls::stream<fuse_vec_in_t>& residual_in,
    hls::stream<fuse_vec_out_t>& fuse_out,
    const ap_int<32> bias[MAX_CHANNELS],
    const ap_int<32> requant_mul[MAX_CHANNELS],
    const ap_int<8> requant_shift[MAX_CHANNELS],
    hls::stream<int>& channels_stream,
    hls::stream<fuse_config_t>& config_stream,
    bool& accel_done
) {
    #pragma HLS INLINE off
    
    // Đọc cấu hình một lần cho mỗi layer
    int channels = channels_stream.read();
    fuse_config_t config = config_stream.read();

    // Buffer nhỏ để hỗ trợ MaxPool 2x2 (line buffer đơn giản)
    static ap_int<8> pool_buf[MAX_WIDTH][FUSE_PARALLEL_SIZE];
    #pragma HLS BIND_STORAGE variable=pool_buf type=ram_2p impl=lutram

    ap_uint<16> c = 0; 
    bool is_last = false;

    process_stream: while (!is_last) {
        #pragma HLS PIPELINE II=1
        
        fuse_vec_in_t in_pkt = conv_in.read();
        fuse_vec_in_t res_pkt;
        
        // 1. Hỗ trợ Residual Add
        if (config.has_residual) {
            res_pkt = residual_in.read();
        }

        fuse_vec_out_t out_pkt;
        is_last = in_pkt.last;

        process_elements: for (int i = 0; i < FUSE_PARALLEL_SIZE; i++) {
            #pragma HLS UNROLL 
            ap_uint<16> current_c = (c + i) % channels; 

            // Phép cộng tích lũy + Residual
            ap_int<32> acc = in_pkt.data[i];
            if (config.has_residual) {
                acc += res_pkt.data[i]; // Cộng nhánh identity
            }

            // Bias & Requant
            ap_int<32> with_bias = acc + bias[current_c];
            ap_int<64> scaled = (ap_int<64>)with_bias * (ap_int<64>)requant_mul[current_c];
            ap_int<32> shifted = (ap_int<32>)(scaled >> requant_shift[current_c]);

            // Saturation 8-bit
            ap_int<8> requantized = (shifted > 127) ? (ap_int<8>)127 : 
                                    (shifted < -128) ? (ap_int<8>)-128 : (ap_int<8>)shifted;

            // Activation (Leaky ReLU)
            ap_int<8> activated = (requantized < 0) ? (ap_int<8>)((requantized * 13) >> 7) : requantized;

            // 2. Xử lý Cat/Slice (Bỏ qua kênh nếu vượt ngưỡng channel_limit)
            if (current_c < config.channel_limit) {
                out_pkt.data[i] = activated;
            } else {
                out_pkt.data[i] = 0; // Hoặc logic bỏ qua gói
            }
            
            // 3. Tích hợp MaxPool (Ví dụ đơn giản cho Pointwise hoặc 2x2 stride 2)
            // Lưu ý: MaxPool 2x2 phức tạp hơn cần thêm logic index hàng/cột
            // Dưới đây là ví dụ minh họa MaxPool 1x2 đơn giản
            /*
            if (config.has_maxpool) {
                 // Logic so sánh với pool_buf và cập nhật...
            }
            */
        }

        out_pkt.last = is_last;
        fuse_out.write(out_pkt);

        c += FUSE_PARALLEL_SIZE;        
        if (c >= channels) c = 0;
    }

    accel_done = true; 
}


void accumulator_top(
    hls::stream<fuse_vec_in_t>& in_mux_data,
    hls::stream<fuse_vec_in_t>& out_fuse_data,
    hls::stream<int>& total_tiles_stream,
    hls::stream<int>& cin_iters_stream 
) {
    #pragma HLS INLINE off
    
    int total_tiles = total_tiles_stream.read();
    int cin_iters = cin_iters_stream.read();

    static ap_int<32> psum_buf[MAX_TILES][FUSE_PARALLEL_SIZE];
    #pragma HLS BIND_STORAGE variable=psum_buf type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=psum_buf type=complete dim=2

    accumulate_cin_loop: for (int c = 0; c < cin_iters; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        
        accumulate_spatial_loop: for (int t = 0; t < total_tiles; t++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
            
            fuse_vec_in_t in_pkt = in_mux_data.read();
            fuse_vec_in_t out_pkt;
            
            process_16_elements: for (int i = 0; i < FUSE_PARALLEL_SIZE; i++) {
                #pragma HLS UNROLL
                ap_int<32> current_val = in_pkt.data[i];
                
                ap_int<32> prev_psum = (c == 0) ? (ap_int<32>)0 : psum_buf[t][i];
                ap_int<32> new_psum = current_val + prev_psum;
                
                psum_buf[t][i] = new_psum;
                
                if (c == cin_iters - 1) {
                    out_pkt.data[i] = new_psum;
                }
            }
            
            if (c == cin_iters - 1) {
                out_pkt.last = in_pkt.last;
                out_fuse_data.write(out_pkt);
            }
        }
    }
}

void compute_to_fuse_serializer(
    hls::stream<psum_block_t>& systolic_in,
    hls::stream<Tile2x2>& winograd_in,
    hls::stream<fuse_vec_in_t>& fuse_out,
    hls::stream<ap_uint<2>>& mode_stream,  // Cập nhật thành ap_uint<2>
    hls::stream<int>& total_elements_stream
) {
    #pragma HLS INLINE off

    // Đọc tĩnh 1 lần
    unsigned int total_elements = total_elements_stream.read();
    ap_uint<2> sel_val = mode_stream.read(); // Đọc giá trị 2-bit

    // Mode 0 là Winograd (như đã quy định ở Scheduler)
    if (sel_val == 0) {
        fuse_vec_in_t out_pkt;
        #pragma HLS ARRAY_PARTITION variable=out_pkt.data type=complete dim=1

        winograd_pack_loop: for (unsigned int i = 0; i < total_elements; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024 
            
            Tile2x2 tile = winograd_in.read(); 
            int w = i % 4; 

            out_pkt.data[w*4 + 0] = tile.data[0][0];
            out_pkt.data[w*4 + 1] = tile.data[0][1];
            out_pkt.data[w*4 + 2] = tile.data[1][0];
            out_pkt.data[w*4 + 3] = tile.data[1][1];

            if (w == 3 || i == (total_elements - 1)) {
                out_pkt.last = (i == (total_elements - 1)); 
                fuse_out.write(out_pkt);
            }
        }
    } else {
        // Mode 1 và Mode 2 đều là Systolic
        systolic_pack_loop: for (unsigned int i = 0; i < total_elements; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024 
            
            psum_block_t block = systolic_in.read(); 
            fuse_vec_in_t out_pkt;
            #pragma HLS ARRAY_PARTITION variable=out_pkt.data type=complete dim=1
            
            systolic_unpack: for (int j = 0; j < FUSE_PARALLEL_SIZE; j++) {
                #pragma HLS UNROLL
                out_pkt.data[j] = block.data[j];
            }
            
            out_pkt.last = (i == (total_elements - 1)); 
            fuse_out.write(out_pkt);
        }
    }
}