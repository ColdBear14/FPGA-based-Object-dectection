#include "TopSystem.h"
#include <iostream>

// void pixel_stream_adapter(hls::stream<pixel_t>& in_ext, hls::stream<pixel_t>& out_int, int total_pixels) {
//     for(int i = 0; i < total_pixels; i++) {
//         #pragma HLS PIPELINE II=1
//         out_int.write(in_ext.read());
//     }
// }

// void weight_stream_adapter(hls::stream<weight_mat_t>& in_ext, hls::stream<weight_mat_t>& out_int, int total_weights) {
//     for(int i = 0; i < total_weights; i++) {
//         #pragma HLS PIPELINE II=1
//         out_int.write(in_ext.read());
//     }
// }

void cnn_accelerator_top(
    hls::stream<pixel_t>& in_pixels,     
    hls::stream<weight_t>& in_weights,
    hls::stream<fuse_vec_out_t>& out_data,

    ap_int<32> *bias_array,
    ap_int<32> *requant_mul_array,
    ap_int<8>  *requant_shift_array,
    
    LayerDescriptor descriptor,
    bool start_accel,
    bool& accel_done
) {
    #pragma HLS INTERFACE axis port=in_pixels
    #pragma HLS INTERFACE axis port=in_weights
    #pragma HLS INTERFACE axis port=out_data

    #pragma HLS INTERFACE m_axi port=bias_array depth=1024 offset=slave bundle=gmem_bias
    #pragma HLS INTERFACE m_axi port=requant_mul_array depth=1024 offset=slave bundle=gmem_req_mul
    #pragma HLS INTERFACE m_axi port=requant_shift_array depth=1024 offset=slave bundle=gmem_req_shift
    
    #pragma HLS INTERFACE s_axilite port=descriptor bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=start_accel bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=accel_done bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    #pragma HLS DATAFLOW

    // Streams nội bộ dữ liệu
    hls::stream<Tile4x4> lb_to_router_stream("lb_to_router");
    hls::stream<weight_mat_t> weight_ram_to_router_stream("weight_ram_to_router");
    hls::stream<Tile4x4> router_to_sys_data("router_to_sys_data");
    hls::stream<Tile4x4> router_to_wino_data("router_to_wino_data");
    hls::stream<weight_mat_t> router_to_sys_weight("router_to_sys_w");
    hls::stream<Tile4x4> router_to_wino_weight("router_to_wino_w");
    hls::stream<psum_block_t> sys_to_mux_stream("sys_to_mux");
    hls::stream<Tile2x2> wino_to_mux_stream("wino_to_mux");
    hls::stream<fuse_vec_in_t> mux_to_fuse_stream("mux_to_fuse");
    hls::stream<fuse_vec_in_t> acc_to_fuse_stream("acc_to_fuse");

    // Streams điều khiển (FSM & Engine Select)
    hls::stream<ap_uint<2>> stream_engine_select_wino("sel_wino");
    hls::stream<ap_uint<2>> stream_engine_select_sys("sel_sys");
    hls::stream<ap_uint<2>> stream_engine_select_lb("sel_lb");
    hls::stream<ap_uint<2>> stream_engine_select_ddemux("sel_ddemux");
    hls::stream<ap_uint<2>> stream_engine_select_wdemux("sel_wdemux");
    hls::stream<ap_uint<2>> stream_engine_select_fuse("sel_fuse");


    // --- CÁC STREAMS CẤU HÌNH MỚI TỪ CONFIG DECODER ---
    hls::stream<int> stream_lb_W("cfg_lb_W");
    hls::stream<int> stream_lb_H("cfg_lb_H");
    hls::stream<int> stream_lb_cin("cfg_lb_cin");

    hls::stream<int> stream_wr_tiles("cfg_wr_tiles");
    hls::stream<int> stream_wr_weights("cfg_wr_weights");
    hls::stream<int> stream_wr_phases("cfg_wr_phases");

    hls::stream<int> stream_cout("cfg_cout");

    hls::stream<int> stream_ddemux_tiles("cfg_ddemux_tiles");
    hls::stream<int> stream_wdemux_weights("cfg_wdemux_weights");
    hls::stream<int> stream_wino_tiles("cfg_wino_tiles");
    hls::stream<int> stream_router2_tiles("cfg_router2_tiles");
    hls::stream<int> stream_acc_packets("cfg_acc_packets");
    hls::stream<int> stream_acc_cin("cfg_acc_cin");
    hls::stream<int> stream_fuse_cout("cfg_fuse_cout");

    hls::stream<int> stream_sys_cin("cfg_sys_cin");
    hls::stream<int> stream_sys_cout("cfg_sys_cout");
    hls::stream<int> stream_sys_tiles_per_ch("cfg_sys_tiles_per_ch");

    hls::stream<int> stream_wino_cin("cfg_wino_cin");
    hls::stream<int> stream_wino_cout("cfg_wino_cout");

    #pragma HLS stream variable=stream_sys_cin depth=2
    #pragma HLS stream variable=stream_sys_cout depth=2
    #pragma HLS stream variable=stream_sys_tiles_per_ch depth=2

    // Khai báo độ sâu cho toàn bộ streams nội bộ
    #pragma HLS stream variable=lb_to_router_stream depth=16
    #pragma HLS stream variable=router_to_sys_data depth=16
    #pragma HLS stream variable=router_to_wino_data depth=16
    #pragma HLS stream variable=router_to_sys_weight depth=16
    #pragma HLS stream variable=router_to_wino_weight depth=16
    #pragma HLS stream variable=sys_to_mux_stream depth=16
    #pragma HLS stream variable=wino_to_mux_stream depth=16
    #pragma HLS stream variable=mux_to_fuse_stream depth=16
    #pragma HLS stream variable=acc_to_fuse_stream depth=16

    // Depth cho config streams (chỉ cần 2 vì ghi 1 lần)
    #pragma HLS stream variable=stream_lb_W depth=2
    #pragma HLS stream variable=stream_lb_H depth=2
    #pragma HLS stream variable=stream_lb_cin depth=2

    #pragma HLS stream variable=stream_cout depth=2
    
    #pragma HLS stream variable=stream_ddemux_tiles depth=2
    #pragma HLS stream variable=stream_wdemux_weights depth=2
    #pragma HLS stream variable=stream_wino_tiles depth=2
    #pragma HLS stream variable=stream_router2_tiles depth=2
    #pragma HLS stream variable=stream_acc_packets depth=2
    #pragma HLS stream variable=stream_acc_cin depth=2
    #pragma HLS stream variable=stream_fuse_cout depth=2

    // Fix for the weight data stream warning (matching your other data streams)
    #pragma HLS stream variable=weight_ram_to_router_stream depth=16

    // Fix for engine select control streams
    #pragma HLS stream variable=stream_engine_select_wino depth=2
    #pragma HLS stream variable=stream_engine_select_sys depth=2
    #pragma HLS stream variable=stream_engine_select_lb depth=2
    #pragma HLS stream variable=stream_engine_select_ddemux depth=2
    #pragma HLS stream variable=stream_engine_select_wdemux depth=2
    #pragma HLS stream variable=stream_engine_select_fuse depth=2

    // Fix for the config decoder -> weight RAM streams
    #pragma HLS stream variable=stream_wr_tiles depth=2
    #pragma HLS stream variable=stream_wr_weights depth=2
    #pragma HLS stream variable=stream_wr_phases depth=2

    // Fix for the config decoder -> winograd streams
    #pragma HLS stream variable=stream_wino_cin depth=2
    #pragma HLS stream variable=stream_wino_cout depth=2

    // ---------------------------------------------------------
    // KẾT NỐI DATAFLOW PIPELINE
    // ---------------------------------------------------------

    // 1. Config Decoder
    std::cout<<"Config Decoder"<<std::endl;
    config_decoder(
        descriptor,
        stream_lb_W, stream_lb_H, stream_lb_cin,
        
        // THÊM: Truyền streams cho Weight RAM vào đây
        stream_wr_tiles, stream_wr_weights, stream_wr_phases,
        
        stream_cout,
        stream_ddemux_tiles, stream_wdemux_weights, stream_router2_tiles,
        stream_sys_cin, stream_sys_cout, stream_sys_tiles_per_ch,
        stream_wino_tiles,
        
        // THÊM: Truyền streams cho Winograd vào đây
        stream_wino_cin, stream_wino_cout,
        
        stream_acc_packets, stream_acc_cin, stream_fuse_cout
    );

    // 2. Scheduler
    std::cout<<"Scheduler"<<std::endl;
    scheduler_top(
        descriptor, start_accel,
        stream_engine_select_sys, stream_engine_select_wino,
        stream_engine_select_lb, stream_engine_select_ddemux, 
        stream_engine_select_wdemux, stream_engine_select_fuse
    );

    // 3. Line Buffer
    std::cout<<"Line Buffer"<<std::endl;
    line_buffer(
        in_pixels, lb_to_router_stream, 
        stream_engine_select_lb, 
        stream_lb_W, stream_lb_H, stream_lb_cin
    );

    std::cout<<"Weight RAM"<<std::endl;
    weight_controller_top(
        in_weights, weight_ram_to_router_stream, 
        stream_wr_tiles, stream_wr_weights, stream_wr_phases
    );

    // 4. Data Router 1 (Phân luồng)
    std::cout<<"Data Router - Data Demux"<<std::endl;
    data_demux(
        lb_to_router_stream, stream_engine_select_ddemux, 
        router_to_sys_data, router_to_wino_data, 
        stream_ddemux_tiles
    );

    std::cout<<"Data Router - Weight Demux"<<std::endl;
    weight_demux(
        weight_ram_to_router_stream , stream_engine_select_wdemux, 
        router_to_sys_weight, router_to_wino_weight, 
        stream_wdemux_weights
    );

    // // 5. Engines
    // std::cout<<"Systolic Engine"<<std::endl;
    // systolic_engine(
    //     router_to_sys_data, router_to_sys_weight, 
    //     3, sys_to_mux_stream, stream_engine_select_sys, 
    //     stream_sys_cin, stream_sys_cout, stream_sys_tiles_per_ch // MỚI
    // );
    
    std::cout<<"WinoGrad Engine"<<std::endl;
    winograd_engine_top(
        router_to_wino_data, router_to_wino_weight, 
        wino_to_mux_stream, stream_engine_select_wino, 
        stream_wino_tiles, stream_wino_cin, stream_wino_cout
    );

    // 6. Data Router 2 (Gom luồng)

    // std::cout<<"Fuse Mux"<<std::endl;
    // compute_to_fuse_serializer(
    //     sys_to_mux_stream, wino_to_mux_stream, mux_to_fuse_stream,
    //     stream_engine_select_fuse, 
    //     stream_router2_tiles
    // );

    // // 7. Accumulator & Fuse
    // std::cout<<"Accumulator"<<std::endl;
    // accumulator_top(
    //     mux_to_fuse_stream, acc_to_fuse_stream, 
    //     stream_acc_packets, stream_acc_cin
    // );

    // std::cout<<"Fuse"<<std::endl;
    // fuse_post_conv(
    //     acc_to_fuse_stream, out_data, 
    //     bias_array, requant_mul_array, requant_shift_array, 
    //     stream_fuse_cout,
    //     accel_done
    // );
}