# 12 MHz clock supplied by serial chip
set_property -dict { PACKAGE_PIN L17   IOSTANDARD LVCMOS33 } [get_ports { sysclk }]; #IO_L12P_T1_MRCC_14 Sch=gclk
create_clock -add -name sys_clk_pin -period 83.33 -waveform {0 41.66} [get_ports {sysclk}];

# Pin 1 to 8, camera rst, pwdn, pclk, xclk, vsync, hsync, scl, sda
set_property -dict { PACKAGE_PIN M3    IOSTANDARD LVCMOS33 } [get_ports { cam_rst }]; #IO_L8N_T1_AD14N_35 Sch=pio[01]
set_property -dict { PACKAGE_PIN L3    IOSTANDARD LVCMOS33 } [get_ports { cam_pwdn }]; #IO_L8P_T1_AD14P_35 Sch=pio[02]
set_property -dict { PACKAGE_PIN A16   IOSTANDARD LVCMOS33 } [get_ports { cam_pclk }]; #IO_L12P_T1_MRCC_16 Sch=pio[03]
set_property -dict { PACKAGE_PIN K3    IOSTANDARD LVCMOS33 } [get_ports { cam_xclk }]; #IO_L7N_T1_AD6N_35 Sch=pio[04]
set_property -dict { PACKAGE_PIN C15   IOSTANDARD LVCMOS33 } [get_ports { cam_vsync }]; #IO_L11P_T1_SRCC_16 Sch=pio[05]
set_property -dict { PACKAGE_PIN H1    IOSTANDARD LVCMOS33 } [get_ports { cam_hsync }]; #IO_L3P_T0_DQS_AD5P_35 Sch=pio[06]
set_property -dict { PACKAGE_PIN A15   IOSTANDARD LVCMOS33 } [get_ports { cam_scl }]; #IO_L6N_T0_VREF_16 Sch=pio[07]
set_property -dict { PACKAGE_PIN B15   IOSTANDARD LVCMOS33 } [get_ports { cam_sda }]; #IO_L11N_T1_SRCC_16 Sch=pio[08]

# Pin 48 downto 41, cam_d[0 TO 7]
set_property -dict { PACKAGE_PIN U5    IOSTANDARD LVCMOS33 } [get_ports { cam_d[7] }]; #IO_L16P_T2_34 Sch=pio[41]
set_property -dict { PACKAGE_PIN U2    IOSTANDARD LVCMOS33 } [get_ports { cam_d[6] }]; #IO_L9N_T1_DQS_34 Sch=pio[42]
set_property -dict { PACKAGE_PIN W6    IOSTANDARD LVCMOS33 } [get_ports { cam_d[5] }]; #IO_L13N_T2_MRCC_34 Sch=pio[43]
set_property -dict { PACKAGE_PIN U3    IOSTANDARD LVCMOS33 } [get_ports { cam_d[4] }]; #IO_L9P_T1_DQS_34 Sch=pio[44]
set_property -dict { PACKAGE_PIN U7    IOSTANDARD LVCMOS33 } [get_ports { cam_d[3] }]; #IO_L19P_T3_34 Sch=pio[45]
set_property -dict { PACKAGE_PIN W7    IOSTANDARD LVCMOS33 } [get_ports { cam_d[2] }]; #IO_L13P_T2_MRCC_34 Sch=pio[46]
set_property -dict { PACKAGE_PIN U8    IOSTANDARD LVCMOS33 } [get_ports { cam_d[1] }]; #IO_L14P_T2_SRCC_34 Sch=pio[47]
set_property -dict { PACKAGE_PIN V8    IOSTANDARD LVCMOS33 } [get_ports { cam_d[0] }]; #IO_L14N_T2_SRCC_34 Sch=pio[48]

# Pmod connector for VGA output
set_property -dict { PACKAGE_PIN G17   IOSTANDARD LVCMOS33 } [get_ports { vga_r[0] }]; #IO_L5N_T0_D07_14 Sch=ja[1]
set_property -dict { PACKAGE_PIN H17   IOSTANDARD LVCMOS33 } [get_ports { vga_r[1] }]; #IO_L5P_T0_D06_14 Sch=ja[7]
set_property -dict { PACKAGE_PIN G19   IOSTANDARD LVCMOS33 } [get_ports { vga_g[0] }]; #IO_L4N_T0_D05_14 Sch=ja[2]
set_property -dict { PACKAGE_PIN H19   IOSTANDARD LVCMOS33 } [get_ports { vga_g[1] }]; #IO_L4P_T0_D04_14 Sch=ja[8]
set_property -dict { PACKAGE_PIN N18   IOSTANDARD LVCMOS33 } [get_ports { vga_b[0] }]; #IO_L9P_T1_DQS_14 Sch=ja[3]
set_property -dict { PACKAGE_PIN J19   IOSTANDARD LVCMOS33 } [get_ports { vga_b[1] }]; #IO_L6N_T0_D08_VREF_14 Sch=ja[9]
set_property -dict { PACKAGE_PIN K18   IOSTANDARD LVCMOS33 } [get_ports { vga_vsync }]; #IO_L8N_T1_D12_14 Sch=ja[10]
set_property -dict { PACKAGE_PIN L18   IOSTANDARD LVCMOS33 } [get_ports { vga_hsync }]; #IO_L8P_T1_D11_14 Sch=ja[4]