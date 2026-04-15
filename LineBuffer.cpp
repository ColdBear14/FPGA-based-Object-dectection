#include "LineBuffer.h"

void line_buffer(
    hls::stream<pixel_t> &in_stream,       
    hls::stream<Tile4x4> &out_tile_stream, 
    hls::stream<ap_uint<2>>& mode_stream,             
    hls::stream<int>& img_width_stream,               
    hls::stream<int>& img_height_stream,
    hls::stream<int>& Cin_stream 
) {
    #pragma HLS INLINE off
    
    int img_width  = img_width_stream.read();
    int img_height = img_height_stream.read();
    int Cin        = Cin_stream.read();
    ap_uint<2> mode = mode_stream.read(); 

    // Thêm chiều thứ 3 để lưu trữ theo Kênh (Channel)
    static pixel_t line_buf[3][MAX_WIDTH][MAX_CIN];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    
    static pixel_t window[4][4][MAX_CIN];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    int col = 0;
    int row = 0;
    const int total_spatial_pixels = img_width * img_height;

    // Lặp theo tọa độ không gian (Pixel)
    process_loop: for (int i = 0; i < total_spatial_pixels; i++) {
        
        // Quét toàn bộ Cin kênh cho 1 tọa độ Pixel
        channel_loop: for (int cin = 0; cin < Cin; cin++) {
            #pragma HLS PIPELINE II=1
            
            pixel_t new_pixel = 0;
            if (!in_stream.empty()) {
                new_pixel = in_stream.read();
            }

            // 1. Shift Window sang trái 
            shift_window_row: for (int r = 0; r < 4; r++) {
                #pragma HLS UNROLL
                shift_window_col: for (int c = 0; c < 3; c++) {
                    #pragma HLS UNROLL
                    window[r][c][cin] = window[r][c + 1][cin];
                }
            }

            // 2. Lấy dữ liệu từ Line Buffer ra Window
            pixel_t b0 = line_buf[0][col][cin];
            pixel_t b1 = line_buf[1][col][cin];
            pixel_t b2 = line_buf[2][col][cin];

            window[0][3][cin] = b0;
            window[1][3][cin] = b1;
            window[2][3][cin] = b2;
            window[3][3][cin] = new_pixel;

            // 3. Cập nhật Line Buffer
            line_buf[0][col][cin] = b1;
            line_buf[1][col][cin] = b2;
            line_buf[2][col][cin] = new_pixel;
        }

        // 4. Xuất Tile 4x4 cho TẤT CẢ các kênh khi đủ Stride
        if (row >= 3 && col >= 3) {
            bool valid_stride = ((col - 3) % 2 == 0) && ((row - 3) % 2 == 0);
            
            if (valid_stride) {
                // Xuất tuần tự từng kênh Cin ra Stream cho Data Router
                out_channel_loop: for(int cin = 0; cin < Cin; cin++) {
                    #pragma HLS PIPELINE II=1
                    Tile4x4 out_tile;
                    copy_tile_row: for (int r = 0; r < 4; r++) {
                        #pragma HLS UNROLL
                        copy_tile_col: for (int c = 0; c < 4; c++) {
                            #pragma HLS UNROLL
                            out_tile.data[r][c] = window[r][c][cin];
                        }
                    }
                    out_tile_stream.write(out_tile);
                }
            }
        }

        // Quản lý tọa độ không gian
        if (col == img_width - 1) {
            col = 0;
            row++;
        } else {
            col++;
        }
    }
}