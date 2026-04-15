#include "tb_winograd.h"
#include "tb_subsystems.h"
#include "tb_top_system.h"
#include "tb_systolic.h"
#include "tb_linebuffer.h"
#include "tb_weightram.h"


int main() {
    std::cout << "--- BẮT ĐẦU VITIS HLS C-SIMULATION ---" << std::endl;
    
    // WinogradEngine_TB();

    // tb_linebuffer();

    // tb_weight_ram();

    // WinogradEngine_DataRouter_TB();

    // LB_WR_Winograd();


    TopSystem_TB();

    // test_mac_systolic_engine();
    
    return 0;
}