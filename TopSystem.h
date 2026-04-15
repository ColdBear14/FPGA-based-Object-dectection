#ifndef TOPSYSTEM_H
#define TOPSYSTEM_H

#include "global.h"
#include "Scheduler.h"
#include "LineBuffer.h"
#include "DataRouter.h"
#include "SystolicEngine.h"
#include "WinogradEngine.h"
#include "Fuse.h"
#include "WeightRAM.h"

void pixel_stream_adapter(hls::stream<pixel_t>& in_ext, hls::stream<pixel_t>& out_int, int total_pixels);
void weight_stream_adapter(hls::stream<weight_mat_t>& in_ext, hls::stream<weight_mat_t>& out_int, int total_weights);

void cnn_accelerator_top(
    // --- Giao tiếp AXI-Stream (Dữ liệu ngoài) ---
    hls::stream<pixel_t>& in_pixels,      // Stream ảnh đầu vào từ DMA
    hls::stream<weight_t>& in_weights,// Stream trọng số từ DMA
    hls::stream<fuse_vec_out_t>& out_data,// Stream kết quả trả về DMA

    ap_int<32> *bias_array,
    ap_int<32> *requant_mul_array,
    ap_int<8>  *requant_shift_array,
    
    // --- Giao tiếp AXI-Lite (Điều khiển từ CPU) ---
    LayerDescriptor descriptor,
    bool start_accel,
    bool& accel_done
);

#endif