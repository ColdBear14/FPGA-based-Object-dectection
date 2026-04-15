#include "helper.h"

void golden_winograd(Tile4x4 d, Tile3x3 g, Tile2x2 &y_ref) {
    // Compute standard convolution: input d (4x4) with kernel v (3x3) -> output y (2x2)
    std::cout << "Output Tile Golden (2x2):" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            data_t sum = 0;
            for (int ki = 0; ki < 3; ++ki) {
                for (int kj = 0; kj < 3; ++kj) {
                    sum += (data_t)(d.data[i+ki][j+kj] * g.data[ki][kj]);
                }
            }
            y_ref.data[i][j] = sum;
            std::cout<<(data_t)y_ref.data[i][j]<<" ";
        }
        std::cout << std::endl;
    }
}

void weight_transform(Tile3x3 g, Tile4x4 &V) {
    #pragma HLS INLINE
    int32_t temp[4][3];

    // Bước 1: temp = G_scaled * g
    for (int j = 0; j < 3; j++) {
        #pragma HLS UNROLL
        int32_t g0 = g.data[0][j];
        int32_t g1 = g.data[1][j];
        int32_t g2 = g.data[2][j];

        temp[0][j] = g0 << 1;           // g0 * 2
        temp[1][j] = (g0 + g1 + g2);    // (g0 + g1 + g2) - Không chia 2
        temp[2][j] = (g0 - g1 + g2);    // (g0 - g1 + g2) - Không chia 2
        temp[3][j] = g2 << 1;           // g2 * 2
    }

    // Bước 2: V = temp * G_scaled^T
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        int32_t t0 = temp[i][0];
        int32_t t1 = temp[i][1];
        int32_t t2 = temp[i][2];

        V.data[i][0] = t0 << 1;         // t0 * 2
        V.data[i][1] = (t0 + t1 + t2);
        V.data[i][2] = (t0 - t1 + t2);
        V.data[i][3] = t2 << 1;         // t2 * 2
    }
}

// MỚI: Tạo Pixel Interleaved (Ưu tiên quét Kênh Cin trước)
void fill_pixel_stream(hls::stream<pixel_t>& stream, int width, int height, int Cin) {
    std::cout << "Khởi tạo Pixel Interleaved (Cin: " << Cin << ")" << std::endl;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for(int cin = 0; cin < Cin; ++cin) {
                // Tạo giá trị giả lập khác nhau giữa các pixel và kênh
                pixel_t pixel = (x + y * width + cin + 1) & 0xFF;
                stream.write(pixel);
            }
        }
    }
}

// MỚI: Weight Stream (Ưu tiên tạo Cin -> Cout để khớp với vòng lặp của Engine)
void fill_weight_stream(hls::stream<weight_t>& stream, int kernel_size, int Cin, int Cout) {
    // Engine đọc: cin=0(cout=0,1..), cin=1(cout=0,1..)
    for (int cin = 0; cin < Cin; ++cin) {
        for (int cout = 0; cout < Cout; ++cout) {
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    weight_t w = (i == j) ? 1 : 0; // Kernel mẫu (Identity)
                    stream.write(w);
                }
            }    
        }
    }
}

// MỚI: Chuyển đổi trọng số Winograd (Khớp thứ tự Cin -> Cout)
void fill_Wino_weight_stream(hls::stream<weight_t>& input, int kernel_size, int Cin, int Cout, hls::stream<weight_t>& output) {
    for (int cin = 0; cin < Cin; ++cin) {
        for (int cout = 0; cout < Cout; ++cout) {
            Tile3x3 g;
            for (int i = 0; i < 3; ++i) 
                for (int j = 0; j < 3; ++j) 
                    g.data[i][j] = input.read();
            
            Tile4x4 V;
            weight_transform(g, V);

            for (int i = 0; i < 4; ++i) 
                for (int j = 0; j < 4; ++j) 
                    output.write(V.data[i][j]);
        }
    }
}

// Hàm đọc kết quả đầu ra
// Hàm đọc kết quả đầu ra (Đọc theo Gói/Packet)
void read_output_stream(hls::stream<fuse_vec_out_t>& stream, int expected_packets) {
    std::cout << "\n--- TOP SYSTEM OUTPUT RESULTS ---" << std::endl;
    int total_pixels_read = 0;
    
    for (int i = 0; i < expected_packets; ++i) {
        if (!stream.empty()) {
            fuse_vec_out_t out_val = stream.read();
            
            std::cout << "Packet [" << i << "]:\n";
            
            // Duyệt qua 16 phần tử (FUSE_PARALLEL_SIZE) trong 1 gói
            for (int j = 0; j < 16; ++j) {
                std::cout << (int)out_val.data[j] << "\t";
                total_pixels_read++;
                
                // Xuống dòng mỗi 4 pixel cho dễ nhìn (tùy chọn)
                if ((j + 1) % 4 == 0) std::cout << "\n";
            }
            
            // In cờ last để debug
            if (out_val.last) {
                std::cout << "-> [Tín hiệu LAST đã được bật]" << std::endl;
            }
            std::cout << "---------------------------------" << std::endl;
            
        } else {
            std::cerr << "LỖI: Output stream bị rỗng sớm ở packet thứ = " << i 
                      << " / " << expected_packets << "!" << std::endl;
            break;
        }
    }
    
    std::cout << ">>> TỔNG SỐ PIXEL ĐÃ ĐỌC: " << total_pixels_read << " <<<" << std::endl;
}