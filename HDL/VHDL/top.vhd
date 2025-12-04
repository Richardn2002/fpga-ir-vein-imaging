LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
LIBRARY UNISIM;
USE UNISIM.vcomponents.ALL;

ENTITY top IS
    PORT (
        -- 12 MHz system clock input
        sysclk : IN STD_LOGIC;

        -- camera power
        cam_rst : OUT STD_LOGIC;
        cam_pwdn : OUT STD_LOGIC;
        cam_xclk : OUT STD_LOGIC;

        -- camera config
        cam_scl : INOUT STD_LOGIC;
        cam_sda : INOUT STD_LOGIC;

        -- camera data
        cam_pclk : IN STD_LOGIC;
        cam_vsync : IN STD_LOGIC;
        cam_hsync : IN STD_LOGIC;
        cam_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        -- vga output
        vga_vsync : OUT STD_LOGIC;
        vga_hsync : OUT STD_LOGIC;
        vga_r : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
        vga_g : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
        vga_b : OUT STD_LOGIC_VECTOR(1 DOWNTO 0)
    );
END top;

ARCHITECTURE arch OF top IS
    SIGNAL core_clk : STD_LOGIC;

    SIGNAL cam_ctrl_clk : STD_LOGIC;
    SIGNAL cam_scl_in : STD_LOGIC;
    SIGNAL cam_scl_out : STD_LOGIC;
    SIGNAL cam_sda_in : STD_LOGIC;
    SIGNAL cam_sda_out : STD_LOGIC;

    SIGNAL vga_pclk : STD_LOGIC;
BEGIN
    -- clock management
    cmt_inst : ENTITY work.cmt
        PORT MAP(
            sysclk => sysclk,
            cam_xclk => cam_xclk,
            cam_ctrl_clk => cam_ctrl_clk,
            vga_clk => vga_pclk,
            core_clk => core_clk
        );
    -- I2C-like SCCB driver
    scl_pin : IOBUF PORT MAP(O => cam_scl_in, IO => cam_scl, I => '0', T => cam_scl_out);
    sda_pin : IOBUF PORT MAP(O => cam_sda_in, IO => cam_sda, I => '0', T => cam_sda_out);
    -- project main
    proj_inst : ENTITY work.proj
        PORT MAP(
            core_clk => core_clk,
            cam_ctrl_clk => cam_ctrl_clk,
            cam_rst => cam_rst,
            cam_pwdn => cam_pwdn,
            cam_scl_in => cam_scl_in,
            cam_scl_out => cam_scl_out,
            cam_sda_in => cam_sda_in,
            cam_sda_out => cam_sda_out,
            cam_pclk => cam_pclk,
            cam_vsync => cam_vsync,
            cam_hsync => cam_hsync,
            cam_d => cam_d,
            vga_pclk => vga_pclk,
            vga_vsync => vga_vsync,
            vga_hsync => vga_hsync,
            vga_r => vga_r,
            vga_g => vga_g,
            vga_b => vga_b
        );
END ARCHITECTURE;