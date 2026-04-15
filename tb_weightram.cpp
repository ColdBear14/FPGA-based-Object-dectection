#include "tb_weightram.h"

void tb_weight_ram() {
    // Khai báo lại các tín hiệu dưới dạng stream theo đúng header
    hls::stream<weight_t> in_weights("in_weights_stream");
    hls::stream<weight_mat_t> out_weight_stream("out_weights_stream");
    hls::stream<int> cfg_num_tiles("cfg_num_tiles_stream");
    hls::stream<int> cfg_weight_size("cfg_weight_size_stream");
    hls::stream<int> cfg_total_phases("cfg_total_phases_stream");

    int test_num_tiles = 3;    
    int test_weight_size = 2;  
    int total_phases = 2;

    cout << ">> [TESTBENCH] Bat dau kiem tra Weight Controller (Stream-based)" << endl;
    cout << "   - So lan Reuse (Spatial Tiles): " << test_num_tiles << endl;
    cout << "   - So vector Weight / Phase: " << test_weight_size << endl;
    cout << "   - Tong so Phase mo phong: " << total_phases << endl;

    cfg_num_tiles.write(test_num_tiles);
    cfg_weight_size.write(test_weight_size);
    cfg_total_phases.write(total_phases);

    cout << "\n>> [TESTBENCH] Bom du lieu vao in_weights stream" << endl;
    for (int phase = 0; phase < total_phases; phase++) {
        for (int w = 0; w < test_weight_size; w++) {
            for (int bank = 0; bank < 16; bank++) {
                // Tạo giá trị giả định (Dummy data) có tính quy luật để dễ debug
                // Công thức: (phase * 100) + (w * 16) + bank
                // Giúp ta nhìn vào giá trị là biết ngay nó thuộc phase nào, vector nào, bank nào
                weight_t val = (weight_t)((phase * 100) + (w * 16) + bank);
                in_weights.write(val);
            }
        }
    }

    cout << "\n>> [TESTBENCH] Dang chay weight_controller_top" << endl;

    weight_controller_top(
        in_weights, 
        out_weight_stream, 
        cfg_num_tiles, 
        cfg_weight_size, 
        cfg_total_phases
    );

    cout << "\n>> [TESTBENCH] Kiem tra luong du lieu dau ra (Co che Reuse)" << endl;

    int pass_count = 0;
    // Tổng số lần đọc mong muốn (mỗi lần đọc ra 1 weight_mat_t chứa 16 phần tử)
    int expected_total_reads = total_phases * test_num_tiles * test_weight_size;
    int actual_reads = 0;

    for (int phase = 0; phase < total_phases; phase++) {
        cout << "--- PHASE " << phase << " ---" << endl;
        
        // Kiểm tra vòng lặp Reuse
        for (int t = 0; t < test_num_tiles; t++) {
            cout << "  Tile (Reuse) thu " << t << ":" << endl;
            
            // Kiểm tra từng vector trong bộ weight của Tile đó
            for (int w = 0; w < test_weight_size; w++) {
                
                if (out_weight_stream.empty()) {
                    cout << "  [LOI FATAL] Stream rong dot ngot tai Phase " << phase 
                         << ", Tile " << t << ", Vector " << w << "!" << endl;
                }

                weight_mat_t out_mat = out_weight_stream.read();
                actual_reads++;
                bool vector_pass = true;

                // Kiểm tra chéo 16 phần tử trong 1 vector so với giá trị đã bơm vào
                for (int bank = 0; bank < 16; bank++) {
                    weight_t expected_val = (weight_t)((phase * 100) + (w * 16) + bank);
                    // Lấy ra 8-bit tương ứng với bank đó
                    weight_t actual_val = out_mat.range(bank * 8 + 7, bank * 8); 

                    if (actual_val != expected_val) {
                        vector_pass = false;
                        cout << "    [LOI DATA] Bank " << bank << " sai. Mong doi: " 
                             << (int)expected_val << ", Thuc te: " << (int)actual_val << endl;
                    }
                }

                if (vector_pass) {
                    cout << "    Vector " << w << ": PASS" << endl;
                    pass_count++;
                }
            }
        }
    }

    cout << ">> TONG KET TESTBENCH:" << endl;
    if (actual_reads == expected_total_reads && pass_count == expected_total_reads) {
        cout << ">> [Thanh cong] TAT CA TEST DEU PASS (" << pass_count << "/" << expected_total_reads << ")" << endl;
        cout << "==============================================" << endl;
    } else {
        cout << ">> [That bai] CO LOI XAY RA! Pass: " << pass_count << ", Expected: " << expected_total_reads << endl;
        cout << "==============================================" << endl;
    }
}