LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY vga_n_core IS
    PORT (
        core_clk : IN STD_LOGIC;
        okay_to_swap : OUT STD_LOGIC;
        trg_from_core : IN STD_LOGIC;

        vga_clk : IN STD_LOGIC;
        ram_reading : IN STD_LOGIC;
        trg_to_vga : OUT STD_LOGIC
    );
END ENTITY;

ARCHITECTURE arch OF vga_n_core IS
    SIGNAL ram_reading_sync : STD_LOGIC;
BEGIN
    cdc_sync_inst : ENTITY work.cdc_sync
        PORT MAP(
            clk_slow => vga_clk,
            sig_from_slow => ram_reading,
            trg_to_slow => trg_to_vga,
            clk_fast => core_clk,
            trg_from_fast => trg_from_core,
            sig_to_fast => ram_reading_sync
        );

    okay_to_swap <= NOT ram_reading_sync;
END ARCHITECTURE;