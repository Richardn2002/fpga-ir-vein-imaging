LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY cam_n_core IS
    PORT (
        cam_clk : IN STD_LOGIC;
        rdy_from_cam : IN STD_LOGIC;
        trg_to_cam : OUT STD_LOGIC;

        core_clk : IN STD_LOGIC;
        rdy_to_core : OUT STD_LOGIC;
        trg_from_core : IN STD_LOGIC
    );
END ENTITY;

ARCHITECTURE arch OF cam_n_core IS
    SIGNAL rdy_from_cam_sync : STD_LOGIC;
    SIGNAL rdy_from_cam_sync_last : STD_LOGIC := '0';
BEGIN
    cdc_sync_inst : ENTITY work.cdc_sync
        PORT MAP(
            clk_slow => cam_clk,
            sig_from_slow => rdy_from_cam,
            trg_to_slow => trg_to_cam,
            clk_fast => core_clk,
            trg_from_fast => trg_from_core,
            sig_to_fast => rdy_from_cam_sync
        );

    PROCESS (core_clk) BEGIN
        IF rising_edge(core_clk) THEN
            rdy_from_cam_sync_last <= rdy_from_cam_sync;
            IF rdy_from_cam_sync_last = '0' AND rdy_from_cam_sync = '1' THEN
                rdy_to_core <= '1';
            ELSE
                rdy_to_core <= '0';
            END IF;
        END IF;
    END PROCESS;
END ARCHITECTURE;