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
    SIGNAL cam_frame_writing : STD_LOGIC;
    SIGNAL cam_frame_writing_to_core : STD_LOGIC;
    SIGNAL cam_ram_swap_trg : STD_LOGIC;
    SIGNAL cam_ram_swap_trg_from_core : STD_LOGIC;

    SIGNAL cam_ram_we : STD_LOGIC;
    SIGNAL cam_x : NATURAL RANGE 0 TO constants.INPUT_X - 1;
    SIGNAL cam_y : NATURAL RANGE 0 TO constants.INPUT_Y - 1;
    SIGNAL cam_ram_d : STD_LOGIC_VECTOR(7 DOWNTO 0);

    SIGNAL cam_ram_we_0 : STD_LOGIC;
    SIGNAL cam_ram_addr_0 : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL cam_ram_d_0 : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL cam_ram_we_1 : STD_LOGIC;
    SIGNAL cam_ram_addr_1 : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL cam_ram_d_1 : STD_LOGIC_VECTOR(7 DOWNTO 0);

    SIGNAL vga_ram_reading : STD_LOGIC;
    SIGNAL vga_okay_to_swap : STD_LOGIC;
    SIGNAL vga_swap_trg_from_core : STD_LOGIC;
    SIGNAL vga_ram_swap_trg : STD_LOGIC;

    SIGNAL vga_ram_we_0 : STD_LOGIC;
    SIGNAL vga_ram_addr_0 : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL vga_ram_d_0 : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL vga_ram_we_1 : STD_LOGIC;
    SIGNAL vga_ram_addr_1 : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL vga_ram_d_1 : STD_LOGIC_VECTOR(7 DOWNTO 0);

    SIGNAL vga_ram_re : STD_LOGIC;
    SIGNAL vga_ram_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL vga_ram_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
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

    flow_ctrl_inst : ENTITY work.flow_ctrl
        PORT MAP(
            core_clk => core_clk,
            cam_frame_writing => cam_frame_writing_to_core,
            cam_ram_swap_trg => cam_ram_swap_trg_from_core,
            vga_okay_to_swap => vga_okay_to_swap,
            vga_ram_swap_trg => vga_swap_trg_from_core
        );

    cam_n_core_inst : ENTITY work.cam_n_core
        PORT MAP(
            cam_clk => cam_pclk,
            frame_writing_from_cam => cam_frame_writing,
            ram_swap_to_cam => cam_ram_swap_trg,
            core_clk => core_clk,
            frame_writing_to_core => cam_frame_writing_to_core,
            ram_swap_from_core => cam_ram_swap_trg_from_core
        );

    cam_vga_inst : ENTITY work.cam_vga
        PORT MAP(
            pclk => cam_pclk,
            vsync => cam_vsync,
            hsync => cam_hsync,
            data => cam_d,
            px_byte => cam_ram_d,
            px_rdy => cam_ram_we,
            frame_writing => cam_frame_writing,
            x => cam_x,
            y => cam_y
        );

    cam_n_bram_inst : ENTITY work.cam_n_bram
        GENERIC MAP(
            INIT_USE_0 => FALSE,
            OUTPUT_X => constants.HESSIAN_OUTPUT_X,
            OUTPUT_Y => constants.HESSIAN_OUTPUT_Y,
            ADDR_BITS => constants.HESSIAN_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            cam_clk => cam_pclk,
            trg => cam_ram_swap_trg,
            cam_we => cam_ram_we,
            cam_x => cam_x,
            cam_y => cam_y,
            cam_d => cam_ram_d,
            bram_we_0 => cam_ram_we_0,
            bram_addr_0 => cam_ram_addr_0,
            bram_d_0 => cam_ram_d_0,
            bram_we_1 => cam_ram_we_1,
            bram_addr_1 => cam_ram_addr_1,
            bram_d_1 => cam_ram_d_1
        );

    bram_to_vga_0 : ENTITY work.bram_tdp
        GENERIC MAP(
            DATA_WIDTH => 8,
            DATA_LEN => constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y,
            ADDR_WIDTH => constants.HESSIAN_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => cam_pclk,
            ce_a => '1',
            we_a => cam_ram_we_0,
            addr_a => cam_ram_addr_0,
            din_a => cam_ram_d_0,
            dout_a => OPEN,
            clk_b => vga_pclk,
            ce_b => '1',
            we_b => vga_ram_we_0,
            addr_b => vga_ram_addr_0,
            din_b => (OTHERS => '0'),
            dout_b => vga_ram_d_0
        );
    bram_to_vga_1 : ENTITY work.bram_tdp
        GENERIC MAP(
            DATA_WIDTH => 8,
            DATA_LEN => constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y,
            ADDR_WIDTH => constants.HESSIAN_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => cam_pclk,
            ce_a => '1',
            we_a => cam_ram_we_1,
            addr_a => cam_ram_addr_1,
            din_a => cam_ram_d_1,
            dout_a => OPEN,
            clk_b => vga_pclk,
            ce_b => '1',
            we_b => vga_ram_we_1,
            addr_b => vga_ram_addr_1,
            din_b => (OTHERS => '0'),
            dout_b => vga_ram_d_1
        );

    vga_n_bram_inst : ENTITY work.vga_n_bram
        GENERIC MAP(
            INIT_USE_0 => TRUE
        )
        PORT MAP(
            vga_clk => vga_pclk,
            trg => vga_ram_swap_trg,
            vga_re => vga_ram_re,
            vga_addr => vga_ram_addr,
            vga_d => vga_ram_d,
            bram_we_0 => vga_ram_we_0,
            bram_addr_0 => vga_ram_addr_0,
            bram_d_0 => vga_ram_d_0,
            bram_we_1 => vga_ram_we_1,
            bram_addr_1 => vga_ram_addr_1,
            bram_d_1 => vga_ram_d_1
        );

    vga_n_core_inst : ENTITY work.vga_n_core
        PORT MAP(
            core_clk => core_clk,
            okay_to_swap => vga_okay_to_swap,
            trg_from_core => vga_swap_trg_from_core,
            vga_clk => vga_pclk,
            ram_reading => vga_ram_reading,
            trg_to_vga => vga_ram_swap_trg
        );

    vga_inst : ENTITY work.vga
        PORT MAP(
            clk => vga_pclk,
            r => vga_r,
            g => vga_g,
            b => vga_b,
            hsync => vga_hsync,
            vsync => vga_vsync,
            ram_re => vga_ram_re,
            ram_addr => vga_ram_addr,
            ram_d => vga_ram_d,
            ram_reading => vga_ram_reading
        );
END ARCHITECTURE;