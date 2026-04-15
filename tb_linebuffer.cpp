#include "tb_linebuffer.h"

void tb_linebuffer() {
    std::cout << "--- BẮT ĐẦU TESTBENCH LINEBUFFER ---" << std::endl;
    hls::stream<pixel_t> in_stream;
    hls::stream<Tile4x4> out_tile_stream;
    hls::stream<ap_uint<2>> mode; // 0: Wino(3x3,s1), 1: Sys(3x3,s2), 2: Sys(1x1,s2)
    hls::stream<int> img_width_stream;
    hls::stream<int> img_height_stream;
    hls::stream<int> Cin_stream;

    Tile4x4 tile;

    std::cout<<"---Wino(3x3,s1)---"<<std::endl;
    mode.write(0);
    img_width_stream.write(8);
    img_height_stream.write(8);
    Cin_stream.write(1);
    fill_pixel_stream(in_stream, 8, 8, 1);
    line_buffer(in_stream, out_tile_stream, mode, img_width_stream, img_height_stream, Cin_stream);
    while (!out_tile_stream.empty()) {
        tile = out_tile_stream.read();
        std::cout << "Tile 4x4:" << std::endl;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                std::cout << tile.data[r][c] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout<<string(50,'-')<<std::endl;

    std::cout<<"---Sys(3x3,s2)---"<<std::endl;
    mode.write(1);
    img_width_stream.write(8);
    img_height_stream.write(8);
    Cin_stream.write(1);
    fill_pixel_stream(in_stream, 8, 8, 1);
    line_buffer(in_stream, out_tile_stream, mode, img_width_stream, img_height_stream, Cin_stream);
    while (!out_tile_stream.empty()) {
        tile = out_tile_stream.read();
        std::cout << "Tile 4x4:" << std::endl;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                std::cout << tile.data[r][c] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout<<string(50,'-')<<std::endl;


    std::cout<<"---Sys(1x1,s2)---"<<std::endl;
    mode.write(2);
    img_width_stream.write(8);
    img_height_stream.write(8);
    Cin_stream.write(1);
    fill_pixel_stream(in_stream, 8, 8, 1);
    line_buffer(in_stream, out_tile_stream, mode, img_width_stream, img_height_stream, Cin_stream);
    while (!out_tile_stream.empty()) {
        tile = out_tile_stream.read();
        std::cout << "Tile 4x4:" << std::endl;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                std::cout << tile.data[r][c] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout<<string(50,'-')<<std::endl;


}