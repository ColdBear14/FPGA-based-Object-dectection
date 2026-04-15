#include "tb_winograd.h"

void WinogradEngine_TB(){
    hls::stream<Tile4x4> in_tile_stream("in_stream");
    hls::stream<Tile4x4> weight_v_stream("weight_stream");
    hls::stream<Tile2x2> out_tile_stream("out_stream");

    hls::stream<ap_uint<2>> mode_stream("mode_stream");
    
    // --- THÊM MỚI: Luồng cấu hình số lượng tile ---
    hls::stream<int> stream_num_tiles("cfg_num_tiles");

    hls::stream<int> stream_wino_cin("cfg_wino_cin");
    hls::stream<int> stream_wino_cout("cfg_wino_cout");

    // 1. Khởi tạo dữ liệu mẫu (Input Tile 4x4)
    Tile4x4 test_d;
    std::cout << "Input Tile D (4x4):" << std::endl;
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            test_d.data[i][j] = (i + j * 4 + 1) & 0xFF;;
            std::cout<<(data_t)test_d.data[i][j]<<" ";
        }
        std::cout << std::endl;
    }

    // 2. Khởi tạo trọng số mẫu
    Tile3x3 test_g;
    std::cout << "Input Tile G (3x3):" << std::endl;
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            test_g.data[i][j] = (data_t)(i == j) ? 1 : 0; 
            std::cout<<(data_t)test_g.data[i][j]<<" ";
        }
        std::cout << std::endl;
    }
    
    Tile4x4 test_v;
    std::cout<< "Weight Transform V(4x4):"<< std::endl;
    weight_transform(test_g, test_v);
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            std::cout<<(data_t)test_v.data[i][j]<<" ";
        }
        std::cout << std::endl;
    }
    
    // 3. Đưa dữ liệu vào Stream
    in_tile_stream.write(test_d);
    weight_v_stream.write(test_v);

    mode_stream.write(0);  // Mode 0 cho Winograd
    
    // Đẩy thông số vào luồng cấu hình trước khi gọi module
    stream_num_tiles.write(1); 
    stream_wino_cin.write(1);
    stream_wino_cout.write(1);


    // 4. Chạy Module HLS (C-Simulation)
    std::cout << "--- Bat dau mo phong Winograd Engine ---" << std::endl;
    winograd_engine_top(in_tile_stream, weight_v_stream, out_tile_stream, mode_stream, stream_num_tiles, stream_wino_cin, stream_wino_cout);

    // 5. Kiểm tra kết quả
    if (!out_tile_stream.empty()) {
        Tile2x2 result = out_tile_stream.read();
        Tile2x2 result_golden;
        
        std::cout << "Ket qua Output Tile (2x2):" << std::endl;
        for(int i=0; i<2; i++) {
            for(int j=0; j<2; j++) {
                std::cout << (data_t)result.data[i][j] << " ";
            }
            std::cout << std::endl;
        }

        // 6. So sánh với giá trị kỳ vọng
        int error = 0;
        
        golden_winograd(test_d, test_g, result_golden);
        for(int i=0; i<2; i++) {
            for(int j=0; j<2; j++) {
                if((data_t)result.data[i][j] != (data_t)result_golden.data[i][j]) error++; 
            }
        }
        std::cout << "LOI: Hai ma tran khac nhau " << error <<std::endl;
        
    } else {
        std::cout << "LOI: Khong co du lieu dau ra!" << std::endl;
    }

    std::cout << "--- Mo phong thanh cong! ---" << std::endl;
}