# core_clk to cam_pclk
set_max_delay 1.5 -datapath_only -from [get_pins proj_inst/cam_n_core_inst/cdc_sync_inst/trg_from_fast_tgl_reg*/C] -to [get_pins proj_inst/cam_n_core_inst/cdc_sync_inst/trg_from_fast_tgl_sync_reg[0]/D]
# core_clk to vga_pclk
set_max_delay 1.5 -datapath_only -from [get_pins proj_inst/vga_n_core_inst/cdc_sync_inst/trg_from_fast_tgl_reg*/C] -to [get_pins proj_inst/vga_n_core_inst/cdc_sync_inst/trg_from_fast_tgl_sync_reg[0]/D]
# cam_pclk to core_clk
set_max_delay 1.5 -datapath_only -from [get_pins proj_inst/cam*inst/rdy_reg*/C] -to [get_pins proj_inst/cam_n_core_inst/cdc_sync_inst/sig_from_slow_sync_reg[0]/D]
# vga_pclk to core_clk
set_max_delay 1.5 -datapath_only -from [get_pins proj_inst/vga_inst/ram_reading_reg*/C] -to [get_pins proj_inst/vga_n_core_inst/cdc_sync_inst/sig_from_slow_sync_reg[0]/D]
