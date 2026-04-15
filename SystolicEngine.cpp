#include "SystolicEngine.h"

// Hàm Input Skew Unit
void apply_input_skew(stream<Tile16x16> &in, pixel_t out_skewed[ARRAY_SIZE]) {
    #pragma HLS INLINE
    static pixel_t skew_regs[ARRAY_SIZE][ARRAY_SIZE]; 
    #pragma HLS ARRAY_PARTITION variable=skew_regs complete dim=0

    // Các biến static để giữ trạng thái unpack giữa các chu kỳ của main_loop
    static Tile16x16 current_tile;
    static ap_uint<4> col_idx = 0; // Tăng lên 4-bit để đếm tới 15
    static bool has_valid_tile = false;

    Column16 col_in;

    // Nếu đã dùng hết cột hoặc chưa có data, đọc Tile mới từ stream
    if (!has_valid_tile && !in.empty()) {
        current_tile = in.read();
        has_valid_tile = true;
        col_idx = 0;
    }

    // Nếu đang có Tile hợp lệ, trích xuất cột hiện tại
    if (has_valid_tile) {
        for (int r = 0; r < ARRAY_SIZE; r++) {
            #pragma HLS UNROLL
            col_in.data[r] = current_tile.data[r][col_idx];
        }
        
        // Chuyển sang cột tiếp theo, nếu là cột cuối thì yêu cầu đọc Tile mới ở chu kỳ sau
        if (col_idx == ARRAY_SIZE - 1) { // Thay đổi linh hoạt theo ARRAY_SIZE
            has_valid_tile = false; 
        } else {
            col_idx++;
        }
    } else {
        // Trạng thái bong bóng (bubble) nếu không có data
        for(int i = 0; i < ARRAY_SIZE; i++) col_in.data[i] = 0;
    }

    // Logic shift của Skew Unit giữ nguyên
    apply_skew_logic: for (int r = 0; r < ARRAY_SIZE; r++) {
        #pragma HLS UNROLL
        for (int i = ARRAY_SIZE - 1; i > 0; i--) {
            skew_regs[r][i] = skew_regs[r][i - 1];
        }
        skew_regs[r][0] = col_in.data[r];
        out_skewed[r] = skew_regs[r][r]; 
    }
}

// Hàm Output Deskew Unit (Không cần sửa đổi logic, sẽ tự loop theo ARRAY_SIZE)
void apply_output_deskew(psum_t paa_outputs[ARRAY_SIZE], stream<psum_block_t> &out) {
    #pragma HLS INLINE
    static psum_t deskew_regs[ARRAY_SIZE][ARRAY_SIZE]; 
    #pragma HLS ARRAY_PARTITION variable=deskew_regs complete dim=0

    static int internal_cycle = 0;

    shift_and_deskew_logic: for (int c = 0; c < ARRAY_SIZE; c++) {
        #pragma HLS UNROLL
        for (int i = ARRAY_SIZE - 1; i > 0; i--) {
            deskew_regs[c][i] = deskew_regs[c][i-1];
        }
        deskew_regs[c][0] = paa_outputs[c];
    }
    
    psum_block_t block_out;
    bool sync_started = (internal_cycle >= ARRAY_SIZE - 1); 

    if (sync_started) {
        sync_block: for (int c = 0; c < ARRAY_SIZE; c++) {
            #pragma HLS UNROLL
            block_out.data[c] = deskew_regs[c][ARRAY_SIZE - 1 - c];
        }
        out.write(block_out);
    }
    internal_cycle++;
}

// Hàm PAA - Post Array Accumulator (Không đổi)
void apply_post_array_accumulator(psum_t psum_row_16[ARRAY_SIZE], psum_t paa_outputs[ARRAY_SIZE], ap_uint<3> ctrl) {
    #pragma HLS INLINE
    static psum_t accumulators[ARRAY_SIZE];
    #pragma HLS ARRAY_PARTITION variable=accumulators complete
    static ap_uint<4> paa_count[ARRAY_SIZE];
    #pragma HLS ARRAY_PARTITION variable=paa_count complete
    
    bool calc_en = ctrl[1];
    bool mode_3x3 = ctrl[0];

    paa_logic_loop: for (int c = 0; c < ARRAY_SIZE; c++) {
        #pragma HLS UNROLL
        if (!calc_en || !mode_3x3) {
            paa_outputs[c] = psum_row_16[c];
            accumulators[c] = 0;
            paa_count[c] = 0;
        } else {
            if (paa_count[c] < KERNEL_SIZE - 1) {
                accumulators[c] += psum_row_16[c];
                paa_outputs[c] = 0;
                paa_count[c]++;
            } else {
                paa_outputs[c] = accumulators[c] + psum_row_16[c];
                accumulators[c] = 0;
                paa_count[c] = 0;
            }
        }
    }
}


// Mô phỏng Systolic Compute (Không đổi, tự scale với ARRAY_SIZE)
void systolic_array_compute_simulation(
    pixel_t skewed_pixels[ARRAY_SIZE],
    weight_t pe_weight_buffer[ARRAY_SIZE][ARRAY_SIZE][KERNEL_SIZE],
    ap_uint<4> pe_rd_ptr[ARRAY_SIZE][ARRAY_SIZE],
    pixel_t pixel_stage[ARRAY_SIZE][ARRAY_SIZE+1], 
    psum_t psum_stage[ARRAY_SIZE+1][ARRAY_SIZE],
    ap_uint<3> ctrl
) {
    #pragma HLS INLINE
    bool calc_en = ctrl[1];
    bool mode_3x3 = ctrl[0];

    update_skewed_inputs: for (int r = 0; r < ARRAY_SIZE; r++) {
        #pragma HLS UNROLL
        pixel_stage[r][0] = skewed_pixels[r];
    }
    init_top_psum_stage: for (int c = 0; c < ARRAY_SIZE; c++) {
        #pragma HLS UNROLL
        psum_stage[0][c] = 0;
    }

    systolic_compute_rows: for (int r = 0; r < ARRAY_SIZE; r++) {
        #pragma HLS UNROLL
        systolic_compute_cols: for (int c = 0; c < ARRAY_SIZE; c++) {
            #pragma HLS UNROLL
            pixel_t p_in = pixel_stage[r][c];
            psum_t ps_in = psum_stage[r][c];
            
            if (calc_en) {
                weight_t current_weight;
                if (mode_3x3) {
                    current_weight = pe_weight_buffer[r][c][pe_rd_ptr[r][c]];
                    if (pe_rd_ptr[r][c] == KERNEL_SIZE - 1) pe_rd_ptr[r][c] = 0;
                    else pe_rd_ptr[r][c]++;
                } else {
                    current_weight = pe_weight_buffer[r][c][0];
                    pe_rd_ptr[r][c] = 0;
                }

                psum_t product = (psum_t)p_in * (psum_t)current_weight;
                psum_t ps_out = ps_in + product;
                pixel_stage[r][c+1] = p_in; 
                psum_stage[r+1][c] = ps_out;
            } else {
                pixel_stage[r][c+1] = p_in;
                psum_stage[r+1][c] = ps_in;
            }
        }
    }
}

// Top Function
void systolic_engine(
    hls::stream<Tile16x16> &pixels_in,
    hls::stream<weight_mat_t> &weights_in, 
    ap_uint<3> ctrl, 
    hls::stream<psum_block_t> &psums_out,
    hls::stream<ap_uint<2>> &mode_stream,
    hls::stream<int> &stream_cin,          
    hls::stream<int> &stream_cout,         
    hls::stream<int> &stream_tiles_per_ch  
) 
{
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE axis port=pixels_in
    #pragma HLS INTERFACE axis port=weights_in
    #pragma HLS INTERFACE axis port=psums_out
    
    static weight_t pe_weight_buffer[ARRAY_SIZE][ARRAY_SIZE][KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=pe_weight_buffer complete dim=0
    static ap_uint<4> pe_rd_ptr[ARRAY_SIZE][ARRAY_SIZE];
    #pragma HLS ARRAY_PARTITION variable=pe_rd_ptr complete dim=0

    static pixel_t pixel_stage[ARRAY_SIZE][ARRAY_SIZE+1];
    #pragma HLS ARRAY_PARTITION variable=pixel_stage complete dim=0
    static psum_t psum_stage[ARRAY_SIZE+1][ARRAY_SIZE];
    #pragma HLS ARRAY_PARTITION variable=psum_stage complete dim=0


    int Cin = stream_cin.read();
    int Cout = stream_cout.read();
    int tiles_per_ch = stream_tiles_per_ch.read();
    ap_uint<2> sel = mode_stream.read();

    if (sel == 1) {
        static bool sim_started = false;
        if (!sim_started) {
            init_r_pixel: for (int r = 0; r < ARRAY_SIZE; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c <= ARRAY_SIZE; c++) {
                    #pragma HLS UNROLL
                    pixel_stage[r][c] = 0;
                }
            }
            init_c_psum: for (int c = 0; c < ARRAY_SIZE; c++) {
                #pragma HLS UNROLL
                for (int r = 0; r <= ARRAY_SIZE; r++) {
                    #pragma HLS UNROLL
                    psum_stage[r][c] = 0;
                }
            }
            reset_ptrs: for (int r = 0; r < ARRAY_SIZE; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c < ARRAY_SIZE; c++) {
                    #pragma HLS UNROLL
                    pe_rd_ptr[r][c] = 0;
                }
            }
            sim_started = true;
        }

        pixel_t skewed_pixels[ARRAY_SIZE];
        psum_t psum_row_16[ARRAY_SIZE]; // Vẫn dùng biến này vì đã định nghĩa [ARRAY_SIZE]
        psum_t paa_outputs[ARRAY_SIZE];

        // --- VÒNG LẶP WEIGHT-STATIONARY CHUẨN ---
        main_loop_cout: for (int cout_idx = 0; cout_idx < Cout; cout_idx++) {
            main_loop_cin: for (int cin_idx = 0; cin_idx < Cin; cin_idx++) {
                
                // 1. Nạp 9 trọng số mới cho bộ lọc hiện tại (ĐỌC THEO TỪNG HÀNG)
                init_k: for (int k = 0; k < KERNEL_SIZE; k++) {
                    init_r: for (int r = 0; r < ARRAY_SIZE; r++) {
                        #pragma HLS PIPELINE II=1
                        // Mỗi chu kỳ đọc 1 hàng (128 bit)
                        weight_mat_t w_row = weights_in.read(); 
                        
                        assign_c: for (int c = 0; c < ARRAY_SIZE; c++) {
                            #pragma HLS UNROLL
                            // Cắt 8-bit tương ứng với cột c để gán vào PE[r][c]
                            pe_weight_buffer[r][c][k] = w_row.range(c * 8 + 7, c * 8);
                        }
                    }
                }

                // 2. Xử lý toàn bộ dữ liệu của Channel này với trọng số vừa nạp
                main_loop_tiles: for (int t = 0; t < tiles_per_ch; t++) {
                    #pragma HLS PIPELINE II=1
                    
                    bool calc_en = ctrl[1];

                    if (calc_en) {
                        apply_input_skew(pixels_in, skewed_pixels); 
                    } else {
                        for (int r = 0; r < ARRAY_SIZE; r++) skewed_pixels[r] = 0;
                    }

                    systolic_array_compute_simulation(skewed_pixels, pe_weight_buffer, pe_rd_ptr, pixel_stage, psum_stage, ctrl);

                    get_psum_row: for (int c = 0; c < ARRAY_SIZE; c++) {
                        psum_row_16[c] = psum_stage[ARRAY_SIZE][c];
                    }

                    apply_post_array_accumulator(psum_row_16, paa_outputs, ctrl);
                    apply_output_deskew(paa_outputs, psums_out);
                }
            }
        }
    }
}