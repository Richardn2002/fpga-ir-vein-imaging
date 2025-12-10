LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY cam_n_core IS
    PORT (
        cam_clk : IN STD_LOGIC;
        frame_writing_from_cam : IN STD_LOGIC;
        ram_swap_to_cam : OUT STD_LOGIC;

        core_clk : IN STD_LOGIC;
        frame_writing_to_core : OUT STD_LOGIC;
        ram_swap_from_core : IN STD_LOGIC
    );
END ENTITY;

ARCHITECTURE arch OF cam_n_core IS
BEGIN
    cdc_sync_inst : ENTITY work.cdc_sync
        PORT MAP(
            clk_slow => cam_clk,
            sig_from_slow => frame_writing_from_cam,
            trg_to_slow => ram_swap_to_cam,
            clk_fast => core_clk,
            trg_from_fast => ram_swap_from_core,
            sig_to_fast => frame_writing_to_core
        );
END ARCHITECTURE;