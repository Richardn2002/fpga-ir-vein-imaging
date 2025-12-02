LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY cam_ov7670_ctrl_TB IS
END cam_ov7670_ctrl_TB;

ARCHITECTURE arch OF cam_ov7670_ctrl_TB IS
    SIGNAL clk : STD_LOGIC;

    SIGNAL rst : STD_LOGIC;
    SIGNAL pwdn : STD_LOGIC;

    SIGNAL scl_in : STD_LOGIC;
    SIGNAL scl_out : STD_LOGIC;
    SIGNAL sda_in : STD_LOGIC;
    SIGNAL sda_out : STD_LOGIC;

    SIGNAL slave_sda : STD_LOGIC;
BEGIN
    cam_ov7670_ctrl_inst : ENTITY work.cam_ov7670_ctrl
        GENERIC MAP(
            IGNORE_I2C_ACK => TRUE
        )
        PORT MAP(
            clk => clk,
            rst => rst,
            pwdn => pwdn,
            scl_in => scl_in,
            scl_out => scl_out,
            sda_in => sda_in,
            sda_out => sda_out
        );

    PROCESS
    BEGIN
        clk <= '0';
        WAIT FOR 41.67 ns;
        clk <= '1';
        WAIT FOR 41.67 ns;
    END PROCESS;
    -- ACK ignored
    slave_sda <= '1';

    scl_in <= scl_out AFTER 1000 ns;
    sda_in <= sda_out AND slave_sda AFTER 1000 ns;
END ARCHITECTURE;