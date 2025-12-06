LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY top_TB IS
END ENTITY;

ARCHITECTURE arch OF top_TB IS
    SIGNAL core_clk : STD_LOGIC;
    SIGNAL cam_ctrl_clk : STD_LOGIC;
    SIGNAL cam_rst : STD_LOGIC;
    SIGNAL cam_pwdn : STD_LOGIC;
    SIGNAL cam_scl_in : STD_LOGIC;
    SIGNAL cam_scl_out : STD_LOGIC;
    SIGNAL cam_sda_in : STD_LOGIC;
    SIGNAL cam_sda_out : STD_LOGIC;
    SIGNAL cam_pclk : STD_LOGIC;
    SIGNAL cam_vsync : STD_LOGIC;
    SIGNAL cam_hsync : STD_LOGIC;
    SIGNAL cam_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL vga_pclk : STD_LOGIC;
    SIGNAL vga_vsync : STD_LOGIC;
    SIGNAL vga_hsync : STD_LOGIC;
    SIGNAL vga : STD_LOGIC_VECTOR(5 DOWNTO 0);
BEGIN
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
            vga_r => vga(5 DOWNTO 4),
            vga_g => vga(3 DOWNTO 2),
            vga_b => vga(1 DOWNTO 0)
        );

    PROCESS
    BEGIN
        -- 240 MHz
        core_clk <= '0';
        WAIT FOR 2.08 ns;
        core_clk <= '1';
        WAIT FOR 2.08 ns;
    END PROCESS;
    PROCESS
    BEGIN
        -- 12 MHz
        cam_ctrl_clk <= '0';
        WAIT FOR 41.67 ns;
        cam_ctrl_clk <= '1';
        WAIT FOR 41.67 ns;
    END PROCESS;
    PROCESS
    BEGIN
        -- 30 MHz
        cam_pclk <= '0';
        WAIT FOR 16.67 ns;
        cam_pclk <= '1';
        WAIT FOR 16.67 ns;
    END PROCESS;
    PROCESS
    BEGIN
        -- 25.153 MHz
        vga_pclk <= '0';
        WAIT FOR 19.88 ns;
        vga_pclk <= '1';
        WAIT FOR 19.88 ns;
    END PROCESS;
END ARCHITECTURE;