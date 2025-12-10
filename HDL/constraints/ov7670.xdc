create_clock -add -name cam_pclk -period 166.66 -waveform {0 83.33} [get_ports {cam_pclk}];
set_input_delay -clock cam_pclk -min 8.00 [get_ports -regexp {cam_d\[.\]|cam_vsync|cam_hsync}]; # t_HD in datasheet
set_input_delay -clock cam_pclk -max 151.66 [get_ports -regexp {cam_d\[.\]|cam_vsync|cam_hsync}]; # t_PCLK - t_SU in datasheet