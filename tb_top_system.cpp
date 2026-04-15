#include "tb_top_system.h"

void TopSystem_TB(){
    hls::stream<pixel_t> in_pixels("in_pixels");
    hls::stream<weight_t> in_weights("in_weights");
    hls::stream<weight_t> in_weights_wino("in_weights_wino");
    hls::stream<fuse_vec_out_t> out_data("out_data");

    // 2. Cấu hình layer descriptor (giả sử một layer convolution đơn giản)
    LayerDescriptor descriptor;
    descriptor.W = 8;          // Chiều rộng ảnh đầu vào
    descriptor.H = 8;          // Chiều cao ảnh đầu vào
    descriptor.Cin = 1;         // Số kênh vào
    descriptor.Cout = 3;        // Số kênh ra
    descriptor.kernel_size = 3;    // Kích thước kernel (3x3)
    descriptor.stride = 1;           // Stride
    descriptor.pad = 0;           // Padding (zero)
    descriptor.preferred_engine = 0;
    
    bool start_accel = false;
    bool accel_done = false;

    // 3. Chuẩn bị dữ liệu đầu vào
    int total_pixels = descriptor.W * descriptor.H * descriptor.Cin;

    fill_pixel_stream(in_pixels, descriptor.W, descriptor.H, descriptor.Cin);

    if(descriptor.preferred_engine == 0) {
        std::cout << "Preparing weights for Winograd engine..." << std::endl;
        fill_weight_stream(in_weights_wino, descriptor.kernel_size, descriptor.Cin, descriptor.Cout);
        fill_Wino_weight_stream(in_weights_wino, descriptor.kernel_size, descriptor.Cin, descriptor.Cout, in_weights);
    } else {
        std::cout << "Preparing weights for Systolic engine..." << std::endl;
        fill_weight_stream(in_weights, descriptor.kernel_size, descriptor.Cin, descriptor.Cout);
    }

    // 4. Kích hoạt accelerator
    start_accel = true;

    ap_int<32> bias_mem[1024];
    ap_int<32> mul_mem[1024];
    ap_int<8>  shift_mem[1024];

    for (int i = 0; i < 1024; i++) {
        bias_mem[i] = 0;      // Không cộng bias
        mul_mem[i] = 1;       // Nhân với 1
        shift_mem[i] = 0;     // Không dịch bit
    }

    // 5. Gọi hàm top (mô phỏng hành vi)
    // Cnn_accelerator_top đã tự động gọi ConfigDecoder bên trong để tạo các luồng tham số,

    cnn_accelerator_top(
        in_pixels, 
        in_weights, 
        out_data, 
        bias_mem,
        mul_mem,
        shift_mem,
        descriptor, 
        start_accel, 
        accel_done
    );

    // // 6. Kiểm tra kết quả
    // if (accel_done) {
    //     std::cout << "Accelerator done!" << std::endl;
    //     // Số lượng phần tử đầu ra: (W' * H' * Cout) với W', H' tính từ convolution
    //     int out_width_expected = (descriptor.W - descriptor.kernel_size + 2*descriptor.pad) / descriptor.stride + 1;
    //     int out_height_expected = (descriptor.H - descriptor.kernel_size + 2*descriptor.pad) / descriptor.stride + 1;
    //     int expected_outputs = out_width_expected * out_height_expected * descriptor.Cout;
    //     int expected_packets = (expected_outputs + 15) / 16; // 36/16 làm tròn lên = 3 gói
    //     read_output_stream(out_data, expected_packets);
    // } else {
    //     std::cerr << "Accelerator did not complete properly!" << std::endl;
    // }
}