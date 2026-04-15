#include "tb_subsystems.h"

void WinogradEngine_DataRouter_TB(){
    std::cout << "\n=== BẮT ĐẦU SIMULATION: WinogradEngine + DataRouter (Testcase 8x8) ===" << std::endl;

    // Streams connecting the modules
    hls::stream<Tile4x4>      data_in_stream("data_in");
    hls::stream<weight_mat_t> weight_in_stream("weight_in");

    hls::stream<ap_uint<2>>   stream_engine_select_ddemux("sel_ddemux");
    hls::stream<ap_uint<2>>   stream_engine_select_wdemux("sel_wdemux");
    hls::stream<ap_uint<2>>   stream_engine_select_wino("sel_wino");

    // Streams inside the dataflow
    hls::stream<Tile4x4>      data_to_winograd("data_to_wino");
    hls::stream<Tile4x4>      data_to_systolic("data_to_sys");
    hls::stream<weight_mat_t> weight_to_systolic("weight_to_sys");
    hls::stream<Tile4x4>      weight_to_winograd("weight_to_wino");
    hls::stream<Tile2x2>      winograd_output("wino_out");

    hls::stream<int> stream_ddemux_tiles("cfg_ddemux_tiles");
    hls::stream<int> stream_wdemux_weights("cfg_wdemux_weights");
    hls::stream<int> stream_wino_tiles("cfg_wino_tiles");

    hls::stream<int> stream_wino_cin("cfg_wino_cin");
    hls::stream<int> stream_wino_cout("cfg_wino_cout");

    // 1. Cấu hình Testcase giống hệt LB_WR_Winograd (Ảnh 8x8)
    int IMG_WIDTH = 8;
    int IMG_HEIGHT = 8;
    int TILES_X = (IMG_WIDTH - 4) / 2 + 1; // = 3
    int TILES_Y = (IMG_HEIGHT - 4) / 2 + 1; // = 3
    int num_tiles = TILES_X * TILES_Y;     // Tổng = 9
    int num_weight_vectors = num_tiles; 

    // Bắn giá trị cấu hình vào streams
    stream_ddemux_tiles.write(num_tiles);
    stream_wdemux_weights.write(num_weight_vectors);
    stream_wino_tiles.write(num_tiles);

    // Select Winograd engine (sel = 0)
    ap_uint<2> sel = 0;
    stream_engine_select_ddemux.write(sel);
    stream_engine_select_wdemux.write(sel);
    stream_engine_select_wino.write(sel);

    stream_wino_cin.write(1);  // Cấu hình số kênh vào (Cin)
    stream_wino_cout.write(1); // Cấu hình số kênh ra (Cout)

    // 2. Tạo dữ liệu đầu vào (Giả lập LineBuffer xuất ra 9 Tiles từ ảnh 8x8)
    data_t img[8][8];
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            img[r][c] = (r + c * 8 + 1) & 0xFF; 
        }
    }

    Tile4x4 input_tiles_golden[9];
    int tile_idx = 0;
    
    for (int r = 3; r < 8; r += 2) {
        for (int c = 3; c < 8; c += 2) {
            Tile4x4 tile;
            for(int tr = 0; tr < 4; tr++){
                for(int tc = 0; tc < 4; tc++){
                    tile.data[tr][tc] = img[r - 3 + tr][c - 3 + tc];
                }
            }
            data_in_stream.write(tile);
            input_tiles_golden[tile_idx++] = tile;
        }
    }

    // 3. TẠO VÀ ĐÓNG GÓI TRỌNG SỐ THÀNH VECTOR 128-BIT (16 PHẦN TỬ)
    Tile3x3 g;  
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            g.data[i][j] = (i == j) ? 1 : 0;

    Tile4x4 V;
    weight_transform(g, V); // Chuyển đổi kernel 3x3 sang dạng Winograd 4x4

    weight_mat_t w_vec; // Vector 128-bit
    
    // Đóng gói 16 phần tử vào vector theo thứ tự idx từ 0 đến 15
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int idx = r * 4 + c;
            // Ép kiểu (int8_t) để giữ đúng dấu trước khi gán vào dải bit
            w_vec.range(idx * 8 + 7, idx * 8) = (int8_t)V.data[r][c];
        }
    }

    // Mô phỏng cơ chế Reuse của WeightRAM: Gửi vector này cho tất cả các Tile
    for(int i = 0; i < num_weight_vectors; i++) {
        weight_in_stream.write(w_vec);
    }

    // 4. Chạy Routers để phân luồng dữ liệu
    data_demux(data_in_stream, stream_engine_select_ddemux, data_to_systolic, data_to_winograd, stream_ddemux_tiles);
    weight_demux(weight_in_stream, stream_engine_select_wdemux, weight_to_systolic, weight_to_winograd, stream_wdemux_weights);

    // 5. Chạy Winograd Engine
    winograd_engine_top(data_to_winograd, weight_to_winograd, winograd_output, stream_engine_select_wino, stream_wino_tiles, stream_wino_cin, stream_wino_cout);

    // 6. Kiểm tra kết quả (Auto-checking)
    int tiles_passed = 0;

    for (int t = 0; t < num_tiles; t++) {
        if (!winograd_output.empty()) {
            Tile2x2 result = winograd_output.read();
            std::cout << "\n--- Winograd output tile " << t << " (2x2) ---\n";
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++)
                    std::cout << (int)result.data[i][j] << " ";
                std::cout << std::endl;
            }

            Tile2x2 golden;
            golden_winograd(input_tiles_golden[t], g, golden);

            bool ok = true;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    if (result.data[i][j] != golden.data[i][j]) {
                        std::cout << "Mismatch at (" << i << "," << j << "): Output="
                                  << (int)result.data[i][j] << " vs Golden=" << (int)golden.data[i][j] << std::endl;
                        ok = false;
                    }
                }
            }
            if (ok) {
                std::cout << "-> Tile " << t << " PASSED.\n";
                tiles_passed++;
            } else {
                std::cout << "-> Tile " << t << " FAILED.\n";
            }
        } else {
            std::cout << "Error: No output from Winograd engine for tile " << t << ".\n";
        }
    }

    if (tiles_passed == num_tiles) {
        std::cout << "\n>>> ALL " << num_tiles << " TILES PASSED SIMULATION! <<<\n";
    } else {
        std::cout << "\n>>> SIMULATION FAILED! Passed " << tiles_passed << "/" << num_tiles << " tiles. <<<\n";
    }
}

void LB_WR_Winograd(){
std::cout << "=== BẮT ĐẦU SIMULATION: LineBuffer + WeightRAM + Winograd ===" << std::endl;

    // 1. KHAI BÁO CÁC HLS STREAMS VÀ TÍN HIỆU ĐIỀU KHIỂN
    hls::stream<pixel_t>      stream_in_pixels("in_pixels");
    hls::stream<weight_t>   stream_in_weights("in_weights");

    hls::stream<weight_t>   stream_in_weights_wino("in_weights_wino");
    
    hls::stream<ap_uint<2>>   stream_engine_select_lb("sel_lb");
    hls::stream<ap_uint<2>>   stream_engine_select_wr("sel_wr");
    hls::stream<ap_uint<2>>   stream_engine_select_ddemux("sel_ddemux");
    hls::stream<ap_uint<2>>   stream_engine_select_wdemux("sel_wdemux");
    hls::stream<ap_uint<2>>   stream_engine_select_wino("sel_wino");

    hls::stream<Tile4x4>      stream_lb_to_router_stream("lb_to_router");

    hls::stream<Tile4x4>      router_to_sys_data("router_to_sys_data");
    hls::stream<Tile4x4>      router_to_wino_data("router_to_wino_data");

    hls::stream<weight_mat_t> router_to_sys_weight("router_to_sys_weight");
    hls::stream<Tile4x4>      router_to_wino_weight("router_to_wino_weight");

    hls::stream<weight_mat_t> stream_wr_to_router_stream("wr_to_router");

    hls::stream<Tile2x2>     wino_to_mux_stream("wino_to_mux");
    hls::stream<int>    stream_wino_tiles("cfg_wino_tiles");

    hls::stream<int> stream_lb_W("cfg_lb_W");
    hls::stream<int> stream_lb_H("cfg_lb_H");
    hls::stream<int> stream_lb_cin("cfg_lb_cin");

    hls::stream<int> stream_num_tiles_wr("cfg_num_tiles");
    hls::stream<int> stream_weight_size_wr("cfg_weight_size");
    hls::stream<int> stream_total_phases_wr("cfg_total_phases");

    hls::stream<int> stream_ddemux_tiles("cfg_ddemux_tiles");
    hls::stream<int> stream_wdemux_weights("cfg_wdemux_weights");

    hls::stream<int> stream_wino_cin("cfg_wino_cin");
    hls::stream<int> stream_wino_cout("cfg_wino_cout");


    // Định nghĩa kích thước test
    int IMG_WIDTH = 8;
    const int IMG_HEIGHT = 8;

    int kernel_size = 3;
    int Cin = 2;
    int Cout = 3;

    int TILES_X = (IMG_WIDTH - 4) / 2 + 1;
    int TILES_Y = (IMG_HEIGHT - 4) / 2 + 1;
    int TOTAL_SPATIAL_TILES = TILES_X * TILES_Y; // = 9

    // --- CÔNG THỨC TOKEN MATCHING QUAN TRỌNG ---
    // DataRouter sẽ phải nhận: 9 Tiles x 2 Cin = 18 Data Tiles
    int total_data_tiles = TOTAL_SPATIAL_TILES * Cin; 
    
    // WeightRAM phải nạp: 2 Cin x 3 Cout = 6 Weight Vectors cho MỖI Tile
    int weight_vectors_per_tile = Cin * Cout;
    
    // DataRouter (Weight) sẽ phân luồng: 9 Tiles x 6 Vectors = 54 Weight Tiles
    int total_weight_vectors = TOTAL_SPATIAL_TILES * weight_vectors_per_tile;

    // Nạp dữ liệu Testbench
    fill_pixel_stream(stream_in_pixels, IMG_WIDTH, IMG_HEIGHT, Cin);
    fill_weight_stream(stream_in_weights, kernel_size, Cin, Cout);
    fill_Wino_weight_stream(stream_in_weights, kernel_size, Cin, Cout, stream_in_weights_wino);

    // Bắn tín hiệu cấu hình
    stream_engine_select_lb.write(0);
    stream_engine_select_wr.write(0);
    stream_engine_select_ddemux.write(0);
    stream_engine_select_wdemux.write(0);
    stream_engine_select_wino.write(0);

    stream_lb_W.write(IMG_WIDTH);
    stream_lb_H.write(IMG_HEIGHT);
    stream_lb_cin.write(Cin);

    // Cấu hình WeightRAM tái sử dụng theo KHÔNG GIAN
    stream_num_tiles_wr.write(TOTAL_SPATIAL_TILES); 
    stream_weight_size_wr.write(weight_vectors_per_tile); // 1 Tile cần 6 Vectors
    stream_total_phases_wr.write(1);

    // Cấu hình Routing Streams (Token trôi qua Demux)
    stream_ddemux_tiles.write(total_data_tiles);       // Bắn 18 Data Tiles cho Winograd
    stream_wdemux_weights.write(total_weight_vectors); // Bắn 54 Weight Tiles cho Winograd

    // Cấu hình Winograd Engine
    stream_wino_tiles.write(TOTAL_SPATIAL_TILES); // Engine sẽ lặp 9 lần Không gian
    stream_wino_cin.write(Cin);
    stream_wino_cout.write(Cout);

    // 2. KHỞI CHẠY MODULES
    line_buffer(stream_in_pixels, stream_lb_to_router_stream, stream_engine_select_lb, stream_lb_W, stream_lb_H, stream_lb_cin);
    
    weight_controller_top(stream_in_weights_wino, stream_wr_to_router_stream, stream_num_tiles_wr, stream_weight_size_wr, stream_total_phases_wr);
    
    data_demux(stream_lb_to_router_stream, stream_engine_select_ddemux, router_to_sys_data, router_to_wino_data, stream_ddemux_tiles);
    
    weight_demux(stream_wr_to_router_stream, stream_engine_select_wdemux, router_to_sys_weight, router_to_wino_weight, stream_wdemux_weights);
    
    winograd_engine_top(router_to_wino_data, router_to_wino_weight, wino_to_mux_stream, stream_engine_select_wino, stream_wino_tiles, stream_wino_cin, stream_wino_cout);

    // 3. KIỂM TRA ĐẦU RA KẾT QUẢ ĐA KÊNH
    std::cout << "\n=== CHECKING WINOGRAD OUTPUT ===" << std::endl;
    for (int t = 0; t < TOTAL_SPATIAL_TILES; t++) {
        std::cout << "\n[ TILE KHÔNG GIAN THỨ " << t << " ]\n";
        for (int cout = 0; cout < Cout; cout++) {
            if (!wino_to_mux_stream.empty()) {
                Tile2x2 result = wino_to_mux_stream.read();
                std::cout << "--- Kênh Output " << cout << " (2x2) ---\n";
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        std::cout << (int)result.data[i][j] << "\t";
                    }
                    std::cout << "\n";
                }
            } else {
                std::cout << "Lỗi: Không có dữ liệu ở kênh " << cout << " của Tile " << t << "\n";
            }
        }
    }
}