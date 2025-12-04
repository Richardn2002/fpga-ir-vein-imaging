LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

USE work.constants;

ENTITY proj IS
    PORT (
        core_clk : IN STD_LOGIC;

        cam_ctrl_clk : IN STD_LOGIC;
        -- camera power
        cam_rst : OUT STD_LOGIC;
        cam_pwdn : OUT STD_LOGIC;
        -- camera config
        cam_scl_in : IN STD_LOGIC;
        cam_scl_out : OUT STD_LOGIC;
        cam_sda_in : IN STD_LOGIC;
        cam_sda_out : OUT STD_LOGIC;
        -- camera data
        cam_pclk : IN STD_LOGIC;
        cam_vsync : IN STD_LOGIC;
        cam_hsync : IN STD_LOGIC;
        cam_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        -- vga output
        vga_pclk : IN STD_LOGIC;
        vga_vsync : OUT STD_LOGIC;
        vga_hsync : OUT STD_LOGIC;
        vga_r : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
        vga_g : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
        vga_b : OUT STD_LOGIC_VECTOR(1 DOWNTO 0)
    );
END proj;

ARCHITECTURE arch OF proj IS
BEGIN
    cam_ov7670_ctrl_inst : ENTITY work.cam_ov7670_ctrl
        PORT MAP(
            clk => cam_ctrl_clk,
            rst => cam_rst,
            pwdn => cam_pwdn,
            scl_in => cam_scl_in,
            scl_out => cam_scl_out,
            sda_in => cam_sda_in,
            sda_out => cam_sda_out
        );

END ARCHITECTURE;
