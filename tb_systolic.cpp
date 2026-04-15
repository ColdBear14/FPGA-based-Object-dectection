#include "tb_systolic.h"

// Hàm test chính
void test_mac_systolic_engine() {
    std::cout << "--- BẮT ĐẦU TEST SYSTOLIC ENGINE ---" << std::endl;

    // 1. Khai báo các Stream giao tiếp
    hls::stream<Tile16x16> pixels_in("pixels_in"); // Đổi thành Tile16x16
    hls::stream<weight_mat_t> weights_in("weights_in");
    hls::stream<psum_block_t> psums_out("psums_out");
    hls::stream<ap_uint<2>> mode("mode");
    hls::stream<int> stream_cin("stream_cin");
    hls::stream<int> stream_cout("stream_cout");
    hls::stream<int> stream_tiles_per_ch("stream_tiles_per_ch");

    // 2. Cấu hình Tham số (Edge case: Test 1 Channel, số lượng tile nhỏ để dễ trace)
    int Cin = 1;
    int Cout = 1;
    int num_tiles = 4; // Số lượng Tile thực tế cần xử lý
    
    // Xử lý xong 'num_tiles', vòng lặp cần chạy (num_tiles * ARRAY_SIZE) chu kỳ.
    int cycles_per_ch = num_tiles * ARRAY_SIZE; 
    
    ap_uint<3> ctrl = 0b011; // bit[1]=calc_en=1, bit[0]=mode_3x3=1

    // Đẩy cấu hình vào stream
    stream_cin.write(Cin);
    stream_cout.write(Cout);
    stream_tiles_per_ch.write(cycles_per_ch); 
    mode.write(1); // Chọn mạch sim 0

    // 3. Khởi tạo Input Data (Quy luật: toàn số 2 để dễ tính MAC)
    for (int t = 0; t < num_tiles; ++t) {
        Tile16x16 tile;
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            for (int j = 0; j < ARRAY_SIZE; ++j) {
                tile.data[i][j] = 2; // Pixel = 2
            }
        }
        pixels_in.write(tile);
    }

    // 4. Khởi tạo Weight Data (Quy luật: toàn số 3 để dễ tính MAC)
    // Đã chuyển sang nạp Row-by-Row. KERNEL_SIZE = 9, mỗi Kernel có ARRAY_SIZE hàng.
    for (int k = 0; k < KERNEL_SIZE; ++k) {
        for (int r = 0; r < ARRAY_SIZE; ++r) {
            weight_mat_t w_row = 0; // Chứa 16 phần tử, tổng 128 bit
            
            // Đóng gói weight cho 1 hàng (16 cột)
            for (int c = 0; c < ARRAY_SIZE; ++c) {
                // Gán giá trị 3 cho mỗi byte trọng số
                w_row.range(c * 8 + 7, c * 8) = 3; 
            }
            
            // Đẩy TỪNG HÀNG vào stream
            weights_in.write(w_row);
        }
    }

    // 5. Gọi hàm Top
    systolic_engine(
        pixels_in, weights_in, ctrl, psums_out, 
        mode, stream_cin, stream_cout, stream_tiles_per_ch
    );

    // 6. Kiểm tra Kết quả (MAC và Số lượng phần tử)

    int expected_outputs = cycles_per_ch - (ARRAY_SIZE - 1);
    
    if (expected_outputs < 0) expected_outputs = 0; // Edge case: số chu kỳ quá ít

    int actual_outputs = 0;
    bool mac_passed = true;

    std::cout << "--- KIỂM TRA ĐẦU RA ---" << std::endl;
    while (!psums_out.empty()) {
        psum_block_t out_block = psums_out.read();
        
        std::cout << "Psum Block [" << actual_outputs << "]: ";
        for (int c = 0; c < ARRAY_SIZE; c++) {
            std::cout << out_block.data[c] << " ";
            
            if (out_block.data[c] != 0 && out_block.data[c] % (2*3) != 0) {
                mac_passed = false;
            }
        }
        std::cout << std::endl;
        actual_outputs++;
    }

    // 7. Report Tổng hợp
    std::cout << "--- KẾT QUẢ TEST ---" << std::endl;
    std::cout << "So luong expected output : " << expected_outputs << std::endl;
    std::cout << "So luong actual output   : " << actual_outputs << std::endl;

    if (actual_outputs == expected_outputs) {
        std::cout << "[PASS] So luong phan tu xuat ra HOAN TOAN KHOP voi pipeline latency!" << std::endl;
    } else {
        std::cout << "[FAIL] So luong phan tu xuat ra KHONG KHOP! (Check lai Skew/Deskew internal cycle)" << std::endl;
    }

    if (mac_passed) {
        std::cout << "[PASS] Phep tinh MAC logic hoat dong dung (ket qua la boi so cua Pixel*Weight)." << std::endl;
    } else {
        std::cout << "[FAIL] Phep tinh MAC sai, co the do logic cong don Psum hoac nạp Weight." << std::endl;
    }
}