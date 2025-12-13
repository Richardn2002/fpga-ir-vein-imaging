LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE std.textio.ALL;
USE IEEE.numeric_std.ALL;

USE work.constants;

ENTITY proj IS
    GENERIC (
        CONSTANT IS_SIM : BOOLEAN := FALSE;
        CONSTANT TEST_INPUT_FILE_0 : STRING := ""
    );
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
    -- CAMERA INPUT
    SIGNAL cam_frame_writing : STD_LOGIC;
    SIGNAL cam_frame_writing_to_core : STD_LOGIC;
    SIGNAL cam_ram_swap_trg : STD_LOGIC;
    SIGNAL cam_ram_swap_trg_from_core : STD_LOGIC;

    SIGNAL cam_ram_we : STD_LOGIC;
    SIGNAL cam_x : NATURAL RANGE 0 TO constants.INPUT_X - 1;
    SIGNAL cam_y : NATURAL RANGE 0 TO constants.INPUT_Y - 1;
    SIGNAL cam_ram_d : STD_LOGIC_VECTOR(7 DOWNTO 0);

    SIGNAL cam_ram_we_0 : STD_LOGIC;
    SIGNAL cam_ram_addr_0 : STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL cam_ram_d_0 : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL cam_ram_we_1 : STD_LOGIC;
    SIGNAL cam_ram_addr_1 : STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL cam_ram_d_1 : STD_LOGIC_VECTOR(7 DOWNTO 0);
    -- CAMERA INPUT END

    -- FLOW CTRL
    SIGNAL clahe_mapping_trg : STD_LOGIC;
    SIGNAL clahe_mapping_rdy : STD_LOGIC;
    SIGNAL clahe_output_trg : STD_LOGIC;
    SIGNAL clahe_output_rdy : STD_LOGIC;
    SIGNAL hessian_conv_r_trg : STD_LOGIC;
    SIGNAL hessian_conv_r_rdy : STD_LOGIC;
    SIGNAL hessian_conv_c_trg : STD_LOGIC;
    SIGNAL hessian_conv_c_rdy : STD_LOGIC;
    SIGNAL hessian_grad_r_trg : STD_LOGIC;
    SIGNAL hessian_grad_r_rdy : STD_LOGIC;
    SIGNAL hessian_grad_c_0_trg : STD_LOGIC;
    SIGNAL hessian_grad_c_0_rdy : STD_LOGIC;
    SIGNAL hessian_grad_rr_cc_trg : STD_LOGIC;
    SIGNAL hessian_grad_rr_cc_rdy : STD_LOGIC;
    SIGNAL hessian_grad_c_1_trg : STD_LOGIC;
    SIGNAL hessian_grad_c_1_rdy : STD_LOGIC;
    SIGNAL hessian_output_trg : STD_LOGIC;
    SIGNAL hessian_output_rdy : STD_LOGIC;

    SIGNAL clahe_input_ram_swap_trg : STD_LOGIC;
    SIGNAL clahe_reader_swap_trg : STD_LOGIC;
    SIGNAL hessian_ram_0_a_user : NATURAL RANGE 0 TO 3;
    SIGNAL hessian_ram_0_b_user : NATURAL RANGE 0 TO 3;
    SIGNAL hessian_ram_1_a_user : NATURAL RANGE 0 TO 3;
    SIGNAL hessian_ram_1_b_user : NATURAL RANGE 0 TO 3;
    SIGNAL hessian_ram_2_a_user : NATURAL RANGE 0 TO 3;
    SIGNAL hessian_ram_2_b_user : NATURAL RANGE 0 TO 3;
    SIGNAL hessian_output_ram_swap_trg : STD_LOGIC;
    -- FLOW CTRL END

    -- MODULE CONNECTIONS
    --- 2x input ram to clahe_input_ram_swapper
    SIGNAL clahe_input_ram_addr_0 : STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL clahe_input_ram_d_0 : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL clahe_input_ram_addr_1 : STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL clahe_input_ram_d_1 : STD_LOGIC_VECTOR(7 DOWNTO 0);
    --- clahe_input_ram_swapper to clahe_n_bram
    SIGNAL clahe_input_re : STD_LOGIC;
    SIGNAL clahe_input_addr : STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL clahe_input_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    --- clahe_n_bram to clahe_mapping and clahe_output
    SIGNAL clahe_mapping_input_re : STD_LOGIC;
    SIGNAL clahe_mapping_input_addr : STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL clahe_mapping_input_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL clahe_output_input_re : STD_LOGIC;
    SIGNAL clahe_output_input_addr : STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL clahe_output_input_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    --- clahe_mapping to bram_hist_mapping
    SIGNAL clahe_mapping_hist_mapping_wen : STD_LOGIC;
    SIGNAL clahe_mapping_hist_mapping_ren : STD_LOGIC;
    SIGNAL clahe_mapping_hist_mapping_we : STD_LOGIC;
    SIGNAL clahe_mapping_hist_mapping_addr : STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS_ALL_PATCHES - 1 DOWNTO 0);
    SIGNAL clahe_mapping_hist_mapping_addr_valid : STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS_ALL_PATCHES - 1 DOWNTO 0);
    SIGNAL clahe_mapping_hist_mapping_din : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL clahe_mapping_hist_mapping_dout : STD_LOGIC_VECTOR(7 DOWNTO 0);
    --- bram_hist_mapping to clahe_output
    SIGNAL clahe_output_hist_mapping_re : STD_LOGIC;
    SIGNAL clahe_output_hist_mapping_addr : STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS_ALL_PATCHES - 1 DOWNTO 0);
    SIGNAL clahe_output_hist_mapping_addr_valid : STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS_ALL_PATCHES - 1 DOWNTO 0);
    SIGNAL clahe_output_hist_mapping_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    --- clahe_output to bram_clahe_output
    SIGNAL clahe_output_clahe_output_we : STD_LOGIC;
    SIGNAL clahe_output_clahe_output_addr : STD_LOGIC_VECTOR(constants.CLAHE_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL clahe_output_clahe_output_addr_valid : STD_LOGIC_VECTOR(constants.CLAHE_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL clahe_output_clahe_output_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    --- hessian_conv_r
    SIGNAL hessian_conv_r_clahe_output_re : STD_LOGIC;
    SIGNAL hessian_conv_r_clahe_output_addr : STD_LOGIC_VECTOR(constants.CLAHE_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_conv_r_clahe_output_addr_valid : STD_LOGIC_VECTOR(constants.CLAHE_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_conv_r_clahe_output_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL hessian_conv_r_ram_0_we : STD_LOGIC;
    SIGNAL hessian_conv_r_ram_0_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_conv_r_ram_0_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    --- hessian_conv_c
    SIGNAL hessian_conv_c_ram_0_re : STD_LOGIC;
    SIGNAL hessian_conv_c_ram_0_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_conv_c_ram_1_we : STD_LOGIC;
    SIGNAL hessian_conv_c_ram_1_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_conv_c_ram_1_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    --- hessian_grad_r
    SIGNAL hessian_grad_r_ram_1_a_re : STD_LOGIC;
    SIGNAL hessian_grad_r_ram_1_a_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_r_ram_1_b_re : STD_LOGIC;
    SIGNAL hessian_grad_r_ram_1_b_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_r_ram_0_we : STD_LOGIC;
    SIGNAL hessian_grad_r_ram_0_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_r_ram_0_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    --- hessian_grad_c_0
    SIGNAL hessian_grad_c_0_ram_1_a_re : STD_LOGIC;
    SIGNAL hessian_grad_c_0_ram_1_a_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_c_0_ram_1_b_re : STD_LOGIC;
    SIGNAL hessian_grad_c_0_ram_1_b_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_c_0_ram_2_we : STD_LOGIC;
    SIGNAL hessian_grad_c_0_ram_2_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_c_0_ram_2_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    --- hessian_grad_rr_cc
    SIGNAL hessian_grad_rr_cc_ram_0_a_re : STD_LOGIC;
    SIGNAL hessian_grad_rr_cc_ram_0_a_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_rr_cc_ram_0_b_re : STD_LOGIC;
    SIGNAL hessian_grad_rr_cc_ram_0_b_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_rr_cc_ram_2_a_re : STD_LOGIC;
    SIGNAL hessian_grad_rr_cc_ram_2_a_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_rr_cc_ram_2_b_re : STD_LOGIC;
    SIGNAL hessian_grad_rr_cc_ram_2_b_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_rr_cc_ram_1_we : STD_LOGIC;
    SIGNAL hessian_grad_rr_cc_ram_1_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_rr_cc_ram_1_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    SIGNAL hessian_grad_rr_cc_ram_3_we : STD_LOGIC;
    SIGNAL hessian_grad_rr_cc_ram_3_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_rr_cc_ram_3_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    --- hessian_grad_c_1
    SIGNAL hessian_grad_c_1_ram_0_a_re : STD_LOGIC;
    SIGNAL hessian_grad_c_1_ram_0_a_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_c_1_ram_0_b_re : STD_LOGIC;
    SIGNAL hessian_grad_c_1_ram_0_b_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_c_1_ram_2_we : STD_LOGIC;
    SIGNAL hessian_grad_c_1_ram_2_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_grad_c_1_ram_2_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    --- hessian_output
    SIGNAL hessian_output_ram_1_re : STD_LOGIC;
    SIGNAL hessian_output_ram_1_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_output_ram_2_re : STD_LOGIC;
    SIGNAL hessian_output_ram_2_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_output_ram_3_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_output_ram_3_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    --- shared outputs of bram_hessian_0/1/2
    SIGNAL ram_0_dout_a : STD_LOGIC_VECTOR(15 DOWNTO 0);
    SIGNAL ram_0_dout_b : STD_LOGIC_VECTOR(15 DOWNTO 0);
    SIGNAL ram_1_dout_a : STD_LOGIC_VECTOR(15 DOWNTO 0);
    SIGNAL ram_1_dout_b : STD_LOGIC_VECTOR(15 DOWNTO 0);
    SIGNAL ram_2_dout_a : STD_LOGIC_VECTOR(15 DOWNTO 0);
    SIGNAL ram_2_dout_b : STD_LOGIC_VECTOR(15 DOWNTO 0);
    --- hessian output to hessian_output_ram_swapper
    SIGNAL hessian_output_we : STD_LOGIC;
    SIGNAL hessian_output_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_output_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    --- hessian_output_ram_swapper to 2x output ram
    SIGNAL hessian_output_ram_we_0 : STD_LOGIC;
    SIGNAL hessian_output_ram_addr_0 : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_output_ram_d_0 : STD_LOGIC_VECTOR(1 DOWNTO 0);
    SIGNAL hessian_output_ram_we_1 : STD_LOGIC;
    SIGNAL hessian_output_ram_addr_1 : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL hessian_output_ram_d_1 : STD_LOGIC_VECTOR(1 DOWNTO 0);
    -- MODULE CONNECTIONS END

    -- VGA OUTPUT
    SIGNAL vga_ram_reading : STD_LOGIC;
    SIGNAL vga_okay_to_swap : STD_LOGIC;
    SIGNAL vga_swap_trg_from_core : STD_LOGIC;
    SIGNAL vga_ram_swap_trg : STD_LOGIC;

    SIGNAL vga_ram_we_0 : STD_LOGIC;
    SIGNAL vga_ram_addr_0 : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL vga_ram_d_0 : STD_LOGIC_VECTOR(1 DOWNTO 0);
    SIGNAL vga_ram_we_1 : STD_LOGIC;
    SIGNAL vga_ram_addr_1 : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL vga_ram_d_1 : STD_LOGIC_VECTOR(1 DOWNTO 0);

    SIGNAL vga_ram_re : STD_LOGIC;
    SIGNAL vga_ram_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL vga_ram_d : STD_LOGIC_VECTOR(7 DOWNTO 0);
    -- VGA OUTPUT END
BEGIN
    cam_ov7670_ctrl_inst : ENTITY work.cam_ov7670_ctrl
        GENERIC MAP(
            IGNORE_I2C_ACK => IS_SIM
        )
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

            clahe_mapping_trg => clahe_mapping_trg,
            clahe_mapping_rdy => clahe_mapping_rdy,
            clahe_output_trg => clahe_output_trg,
            clahe_output_rdy => clahe_output_rdy,
            hessian_conv_r_trg => hessian_conv_r_trg,
            hessian_conv_r_rdy => hessian_conv_r_rdy,
            hessian_conv_c_trg => hessian_conv_c_trg,
            hessian_conv_c_rdy => hessian_conv_c_rdy,
            hessian_grad_r_trg => hessian_grad_r_trg,
            hessian_grad_r_rdy => hessian_grad_r_rdy,
            hessian_grad_c_0_trg => hessian_grad_c_0_trg,
            hessian_grad_c_0_rdy => hessian_grad_c_0_rdy,
            hessian_grad_rr_cc_trg => hessian_grad_rr_cc_trg,
            hessian_grad_rr_cc_rdy => hessian_grad_rr_cc_rdy,
            hessian_grad_c_1_trg => hessian_grad_c_1_trg,
            hessian_grad_c_1_rdy => hessian_grad_c_1_rdy,
            hessian_output_trg => hessian_output_trg,
            hessian_output_rdy => hessian_output_rdy,

            clahe_reader_swap_trg => clahe_reader_swap_trg,
            hessian_ram_0_a_user => hessian_ram_0_a_user,
            hessian_ram_0_b_user => hessian_ram_0_b_user,
            hessian_ram_1_a_user => hessian_ram_1_a_user,
            hessian_ram_1_b_user => hessian_ram_1_b_user,
            hessian_ram_2_a_user => hessian_ram_2_a_user,
            hessian_ram_2_b_user => hessian_ram_2_b_user,

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

    -- for some reason vivado does not like if generate without a label
    real_cam_input : IF NOT IS_SIM GENERATE
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
    END GENERATE real_cam_input;
    -- for some reason the formatter does not like if generate else generate
    test_cam_input : IF IS_SIM GENERATE
        PROCESS
            FILE img_file : text OPEN READ_MODE IS TEST_INPUT_FILE_0;
            VARIABLE ln : line;
            VARIABLE value : NATURAL;
            VARIABLE idx : NATURAL := 0;
            TYPE input_t IS ARRAY (0 TO constants.INPUT_X * constants.INPUT_Y - 1) OF NATURAL;
            VARIABLE test_input_0 : input_t;

            VARIABLE px_idx : NATURAL;
            VARIABLE frame_interval_cnt : NATURAL;
        BEGIN
            WHILE NOT endfile(img_file) LOOP
                readline(img_file, ln);
                read(ln, value);
                IF idx < constants.INPUT_X * constants.INPUT_Y THEN
                    test_input_0(idx) := value;
                ELSE
                    ASSERT false
                    REPORT "File has more data than expected input."
                        SEVERITY error;
                END IF;
                idx := idx + 1;
            END LOOP;
            IF idx /= constants.INPUT_X * constants.INPUT_Y THEN
                ASSERT false
                REPORT "File has less data than expected input."
                    SEVERITY error;
            END IF;

            cam_ram_d <= (OTHERS => '0');
            cam_ram_we <= '0';
            cam_frame_writing <= '0';
            cam_x <= 0;
            cam_y <= 0;
            WAIT FOR 10 ns;

            WHILE TRUE LOOP
                px_idx := 0;
                frame_interval_cnt := 1000;

                WHILE px_idx /= constants.INPUT_X * constants.INPUT_Y LOOP
                    WAIT UNTIL rising_edge(cam_pclk);

                    cam_ram_d <= STD_LOGIC_VECTOR(to_unsigned(test_input_0(px_idx), 8));
                    cam_ram_we <= '1';
                    cam_frame_writing <= '1';
                    cam_x <= px_idx MOD constants.INPUT_X;
                    cam_y <= px_idx / constants.INPUT_X;

                    px_idx := px_idx + 1;
                END LOOP;

                cam_frame_writing <= '0';

                WHILE frame_interval_cnt > 0 LOOP
                    WAIT UNTIL rising_edge(cam_pclk);
                    frame_interval_cnt := frame_interval_cnt - 1;
                END LOOP;
            END LOOP;
        END PROCESS;
    END GENERATE test_cam_input;

    cam_n_bram_inst : ENTITY work.cam_n_bram
        GENERIC MAP(
            INIT_USE_0 => FALSE
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

    bram_input_0 : ENTITY work.bram_tdp
        GENERIC MAP(
            DATA_WIDTH => 8,
            DATA_LEN => constants.INPUT_X * constants.INPUT_Y,
            ADDR_WIDTH => constants.INPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => cam_pclk,
            ce_a => '1',
            we_a => cam_ram_we_0,
            addr_a => cam_ram_addr_0,
            din_a => cam_ram_d_0,
            dout_a => OPEN,
            clk_b => core_clk,
            ce_b => '1',
            we_b => '0',
            addr_b => clahe_input_ram_addr_0,
            din_b => (OTHERS => '0'),
            dout_b => clahe_input_ram_d_0
        );

    bram_input_1 : ENTITY work.bram_tdp
        GENERIC MAP(
            DATA_WIDTH => 8,
            DATA_LEN => constants.INPUT_X * constants.INPUT_Y,
            ADDR_WIDTH => constants.INPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => cam_pclk,
            ce_a => '1',
            we_a => cam_ram_we_1,
            addr_a => cam_ram_addr_1,
            din_a => cam_ram_d_1,
            dout_a => OPEN,
            clk_b => core_clk,
            ce_b => '1',
            we_b => '0',
            addr_b => clahe_input_ram_addr_1,
            din_b => (OTHERS => '0'),
            dout_b => clahe_input_ram_d_1
        );

    -- swap together with camera, but always point to different ram
    clahe_input_ram_swap_trg <= cam_ram_swap_trg_from_core;
    clahe_input_ram_swapper : ENTITY work.bram_swapper_r
        GENERIC MAP(
            INIT_USE_0 => TRUE,
            ADDR_BITS => constants.INPUT_ADDR_BITS,
            DATA_BITS => 8
        )
        PORT MAP(
            clk => core_clk,
            trg => clahe_input_ram_swap_trg,
            re => clahe_input_re,
            addr => clahe_input_addr,
            d => clahe_input_d,
            bram_we_0 => OPEN,
            bram_addr_0 => clahe_input_ram_addr_0,
            bram_d_0 => clahe_input_ram_d_0,
            bram_we_1 => OPEN,
            bram_addr_1 => clahe_input_ram_addr_1,
            bram_d_1 => clahe_input_ram_d_1
        );

    clahe_n_bram_inst : ENTITY work.clahe_n_bram
        GENERIC MAP(
            -- flow ctrl will issue a pulse on entering first stage, swapping to mapping correctly
            -- on entering second stage it will issue another pulse, swapping back
            INIT_READER_MAPPING => FALSE
        )
        PORT MAP(
            core_clk => core_clk,
            clahe_reader_swap_trg => clahe_reader_swap_trg,
            bram_re => clahe_input_re,
            bram_addr => clahe_input_addr,
            bram_d => clahe_input_d,
            clahe_mapping_re => clahe_mapping_input_re,
            clahe_mapping_addr => clahe_mapping_input_addr,
            clahe_mapping_d => clahe_mapping_input_d,
            clahe_output_re => clahe_output_input_re,
            clahe_output_addr => clahe_output_input_addr,
            clahe_output_d => clahe_output_input_d
        );

    clahe_mapping_hist_mapping_we <= clahe_mapping_hist_mapping_wen;
    clahe_mapping_hist_mapping_addr_valid <= clahe_mapping_hist_mapping_addr WHEN (clahe_mapping_hist_mapping_wen OR clahe_mapping_hist_mapping_ren) = '1'
        ELSE
        (OTHERS => '0');
    clahe_mapping : ENTITY work.CLAHE_controller
        PORT MAP(
            clk => core_clk,
            start => clahe_mapping_trg,
            done => clahe_mapping_rdy,
            img_in_en => clahe_mapping_input_re,
            img_in_addr => clahe_mapping_input_addr,
            img_in_d => clahe_mapping_input_d,
            mapping_ren => clahe_mapping_hist_mapping_ren,
            mapping_wen => clahe_mapping_hist_mapping_wen,
            mapping_addr => clahe_mapping_hist_mapping_addr,
            mapping_din => clahe_mapping_hist_mapping_din,
            mapping_dout => clahe_mapping_hist_mapping_dout
        );

    bram_hist_mapping : ENTITY work.bram_tdp
        GENERIC MAP(
            DATA_WIDTH => 8,
            DATA_LEN => constants.CLAHE_MAPPING_LEN_ALL_PATCHES,
            ADDR_WIDTH => constants.CLAHE_MAPPING_ADDR_BITS_ALL_PATCHES
        )
        PORT MAP(
            clk_a => core_clk,
            ce_a => '1',
            we_a => clahe_mapping_hist_mapping_we,
            addr_a => clahe_mapping_hist_mapping_addr_valid,
            din_a => clahe_mapping_hist_mapping_dout,
            dout_a => clahe_mapping_hist_mapping_din,
            clk_b => core_clk,
            ce_b => '1',
            we_b => '0',
            addr_b => clahe_output_hist_mapping_addr_valid,
            din_b => (OTHERS => '0'),
            dout_b => clahe_output_hist_mapping_d
        );

    clahe_output_hist_mapping_addr_valid <= clahe_output_hist_mapping_addr WHEN clahe_output_hist_mapping_re = '1'
        ELSE
        (OTHERS => '0');
    clahe_output : ENTITY work.CLAHE_output
        PORT MAP(
            clk => core_clk,
            trg => clahe_output_trg,
            rdy => clahe_output_rdy,
            img_in_en => clahe_output_input_re,
            img_in_addr => clahe_output_input_addr,
            img_in_d => clahe_output_input_d,
            mapping_in_en => clahe_output_hist_mapping_re,
            mapping_in_addr => clahe_output_hist_mapping_addr,
            mapping_in_d => clahe_output_hist_mapping_d,
            clahe_out_en => clahe_output_clahe_output_we,
            clahe_out_addr => clahe_output_clahe_output_addr,
            clahe_out_d => clahe_output_clahe_output_d
        );
    clahe_output_clahe_output_addr_valid <= clahe_output_clahe_output_addr WHEN clahe_output_clahe_output_we = '1'
        ELSE
        (OTHERS => '0');

    bram_clahe_output : ENTITY work.bram_tdp
        GENERIC MAP(
            DATA_WIDTH => 8,
            DATA_LEN => constants.CLAHE_OUTPUT_LEN,
            ADDR_WIDTH => constants.CLAHE_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => core_clk,
            ce_a => '1',
            we_a => clahe_output_clahe_output_we,
            addr_a => clahe_output_clahe_output_addr_valid,
            din_a => clahe_output_clahe_output_d,
            dout_a => OPEN,
            clk_b => core_clk,
            ce_b => '1',
            we_b => '0',
            addr_b => hessian_conv_r_clahe_output_addr_valid,
            din_b => (OTHERS => '0'),
            dout_b => hessian_conv_r_clahe_output_d
        );

    hessian_conv_r_clahe_output_addr_valid <= hessian_conv_r_clahe_output_addr WHEN hessian_conv_r_clahe_output_re = '1'
        ELSE
        (OTHERS => '0');
    hessian_conv_r : ENTITY work.HESSIAN_conv_r
        PORT MAP(
            clk => core_clk,
            trg => hessian_conv_r_trg,
            rdy => hessian_conv_r_rdy,
            clahe_in_en => hessian_conv_r_clahe_output_re,
            clahe_in_addr => hessian_conv_r_clahe_output_addr,
            clahe_in_d => hessian_conv_r_clahe_output_d,
            conv_out_en => hessian_conv_r_ram_0_we,
            conv_out_addr => hessian_conv_r_ram_0_addr,
            conv_out_d => hessian_conv_r_ram_0_d
        );

    hessian_conv_c : ENTITY work.HESSIAN_conv_c
        PORT MAP(
            clk => core_clk,
            trg => hessian_conv_c_trg,
            rdy => hessian_conv_c_rdy,
            conv_in_en => hessian_conv_c_ram_0_re,
            conv_in_addr => hessian_conv_c_ram_0_addr,
            conv_in_d => ram_0_dout_a,
            conv_out_en => hessian_conv_c_ram_1_we,
            conv_out_addr => hessian_conv_c_ram_1_addr,
            conv_out_d => hessian_conv_c_ram_1_d
        );

    hessian_grad_r_ram_1_a_re <= '1';
    hessian_grad_r_ram_1_b_re <= '1';
    hessian_grad_r : ENTITY work.hessian_grad_r
        PORT MAP(
            clk => core_clk,
            start => hessian_grad_r_trg,
            done => hessian_grad_r_rdy,
            conv0_addr => hessian_grad_r_ram_1_a_addr,
            conv1_addr => hessian_grad_r_ram_1_b_addr,
            conv0_dout => ram_1_dout_a,
            conv1_dout => ram_1_dout_b,
            gr_addr => hessian_grad_r_ram_0_addr,
            gr_din => hessian_grad_r_ram_0_d,
            gr_we => hessian_grad_r_ram_0_we
        );

    hessian_grad_c_0_ram_1_a_re <= '1';
    hessian_grad_c_0_ram_1_b_re <= '1';
    hessian_grad_c_0 : ENTITY work.hessian_grad_c
        PORT MAP(
            clk => core_clk,
            start => hessian_grad_c_0_trg,
            done => hessian_grad_c_0_rdy,
            conv0_addr => hessian_grad_c_0_ram_1_a_addr,
            conv1_addr => hessian_grad_c_0_ram_1_b_addr,
            conv0_dout => ram_1_dout_a,
            conv1_dout => ram_1_dout_b,
            gc_addr => hessian_grad_c_0_ram_2_addr,
            gc_din => hessian_grad_c_0_ram_2_d,
            gc_we => hessian_grad_c_0_ram_2_we
        );

    hessian_grad_rr_cc_ram_0_a_re <= '1';
    hessian_grad_rr_cc_ram_0_b_re <= '1';
    hessian_grad_rr_cc_ram_2_a_re <= '1';
    hessian_grad_rr_cc_ram_2_b_re <= '1';
    hessian_grad_rr_cc : ENTITY work.hessian_grad_rr_cc
        PORT MAP(
            clk => core_clk,
            start => hessian_grad_rr_cc_trg,
            done => hessian_grad_rr_cc_rdy,
            gr0_addr => hessian_grad_rr_cc_ram_0_a_addr,
            gr0_dout => ram_0_dout_a,
            gr1_addr => hessian_grad_rr_cc_ram_0_b_addr,
            gr1_dout => ram_0_dout_b,
            gc0_addr => hessian_grad_rr_cc_ram_2_a_addr,
            gc0_dout => ram_2_dout_a,
            gc1_addr => hessian_grad_rr_cc_ram_2_b_addr,
            gc1_dout => ram_2_dout_b,
            rr_p_cc_addr => hessian_grad_rr_cc_ram_1_addr,
            rr_p_cc_din => hessian_grad_rr_cc_ram_1_d,
            rr_p_cc_we => hessian_grad_rr_cc_ram_1_we,
            rr_m_cc_addr => hessian_grad_rr_cc_ram_3_addr,
            rr_m_cc_din => hessian_grad_rr_cc_ram_3_d,
            rr_m_cc_we => hessian_grad_rr_cc_ram_3_we
        );

    hessian_grad_c_1_ram_0_a_re <= '1';
    hessian_grad_c_1_ram_0_b_re <= '1';
    hessian_grad_c_1 : ENTITY work.hessian_grad_c
        PORT MAP(
            clk => core_clk,
            start => hessian_grad_c_1_trg,
            done => hessian_grad_c_1_rdy,
            conv0_addr => hessian_grad_c_1_ram_0_a_addr,
            conv1_addr => hessian_grad_c_1_ram_0_b_addr,
            conv0_dout => ram_0_dout_a,
            conv1_dout => ram_0_dout_b,
            gc_addr => hessian_grad_c_1_ram_2_addr,
            gc_din => hessian_grad_c_1_ram_2_d,
            gc_we => hessian_grad_c_1_ram_2_we
        );

    hessian_output_ram_1_re <= '1';
    hessian_output_ram_2_re <= '1';
    hessian_output : ENTITY work.hessian_output
        PORT MAP(
            clk => core_clk,
            trg => hessian_output_trg,
            rdy => hessian_output_rdy,
            rrp_addr => hessian_output_ram_1_addr,
            rrp_dout => ram_1_dout_a,
            rrm_addr => hessian_output_ram_3_addr,
            rrm_dout => hessian_output_ram_3_d,
            rc_addr => hessian_output_ram_2_addr,
            rc_dout => ram_2_dout_a,
            out_addr => hessian_output_addr,
            out_din => hessian_output_d,
            out_we => hessian_output_we
        );

    bram_hessian_0 : ENTITY work.bram_multi_rw
        GENERIC MAP(
            DATA_WIDTH => 16,
            DATA_LEN => constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y,
            ADDR_WIDTH => constants.HESSIAN_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => core_clk,
            sel_a => hessian_ram_0_a_user,
            we_a_0 => '0',
            re_a_0 => hessian_conv_c_ram_0_re,
            addr_a_0 => hessian_conv_c_ram_0_addr,
            din_a_0 => (OTHERS => '0'),
            we_a_1 => '0',
            re_a_1 => hessian_grad_c_1_ram_0_a_re,
            addr_a_1 => hessian_grad_c_1_ram_0_a_addr,
            din_a_1 => (OTHERS => '0'),
            we_a_2 => '0',
            re_a_2 => hessian_grad_rr_cc_ram_0_a_re,
            addr_a_2 => hessian_grad_rr_cc_ram_0_a_addr,
            din_a_2 => (OTHERS => '0'),
            we_a_3 => '0',
            re_a_3 => '0',
            addr_a_3 => (OTHERS => '0'),
            din_a_3 => (OTHERS => '0'),
            dout_a => ram_0_dout_a,
            clk_b => core_clk,
            sel_b => hessian_ram_0_b_user,
            we_b_0 => hessian_conv_r_ram_0_we,
            re_b_0 => '0',
            addr_b_0 => hessian_conv_r_ram_0_addr,
            din_b_0 => hessian_conv_r_ram_0_d,
            we_b_1 => hessian_grad_r_ram_0_we,
            re_b_1 => '0',
            addr_b_1 => hessian_grad_r_ram_0_addr,
            din_b_1 => hessian_grad_r_ram_0_d,
            we_b_2 => '0',
            re_b_2 => hessian_grad_c_1_ram_0_b_re,
            addr_b_2 => hessian_grad_c_1_ram_0_b_addr,
            din_b_2 => (OTHERS => '0'),
            we_b_3 => '0',
            re_b_3 => hessian_grad_rr_cc_ram_0_b_re,
            addr_b_3 => hessian_grad_rr_cc_ram_0_b_addr,
            din_b_3 => (OTHERS => '0'),
            dout_b => ram_0_dout_b
        );

    bram_hessian_1 : ENTITY work.bram_multi_rw
        GENERIC MAP(
            DATA_WIDTH => 16,
            DATA_LEN => constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y,
            ADDR_WIDTH => constants.HESSIAN_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => core_clk,
            sel_a => hessian_ram_1_a_user,
            we_a_0 => '0',
            re_a_0 => hessian_grad_r_ram_1_a_re,
            addr_a_0 => hessian_grad_r_ram_1_a_addr,
            din_a_0 => (OTHERS => '0'),
            we_a_1 => '0',
            re_a_1 => hessian_grad_c_0_ram_1_a_re,
            addr_a_1 => hessian_grad_c_0_ram_1_a_addr,
            din_a_1 => (OTHERS => '0'),
            we_a_2 => '0',
            re_a_2 => hessian_output_ram_1_re,
            addr_a_2 => hessian_output_ram_1_addr,
            din_a_2 => (OTHERS => '0'),
            we_a_3 => '0',
            re_a_3 => '0',
            addr_a_3 => (OTHERS => '0'),
            din_a_3 => (OTHERS => '0'),
            dout_a => ram_1_dout_a,
            clk_b => core_clk,
            sel_b => hessian_ram_1_b_user,
            we_b_0 => hessian_conv_c_ram_1_we,
            re_b_0 => '0',
            addr_b_0 => hessian_conv_c_ram_1_addr,
            din_b_0 => hessian_conv_c_ram_1_d,
            we_b_1 => '0',
            re_b_1 => hessian_grad_r_ram_1_b_re,
            addr_b_1 => hessian_grad_r_ram_1_b_addr,
            din_b_1 => (OTHERS => '0'),
            we_b_2 => hessian_grad_rr_cc_ram_1_we,
            re_b_2 => '0',
            addr_b_2 => hessian_grad_rr_cc_ram_1_addr,
            din_b_2 => hessian_grad_rr_cc_ram_1_d,
            we_b_3 => '0',
            re_b_3 => hessian_grad_c_0_ram_1_b_re,
            addr_b_3 => hessian_grad_c_0_ram_1_b_addr,
            din_b_3 => (OTHERS => '0'),
            dout_b => ram_1_dout_b
        );

    bram_hessian_2 : ENTITY work.bram_multi_rw
        GENERIC MAP(
            DATA_WIDTH => 16,
            DATA_LEN => constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y,
            ADDR_WIDTH => constants.HESSIAN_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => core_clk,
            sel_a => hessian_ram_2_a_user,
            we_a_0 => hessian_grad_c_1_ram_2_we,
            re_a_0 => '0',
            addr_a_0 => hessian_grad_c_1_ram_2_addr,
            din_a_0 => hessian_grad_c_1_ram_2_d,
            we_a_1 => '0',
            re_a_1 => hessian_grad_rr_cc_ram_2_a_re,
            addr_a_1 => hessian_grad_rr_cc_ram_2_a_addr,
            din_a_1 => (OTHERS => '0'),
            we_a_2 => hessian_grad_c_0_ram_2_we,
            re_a_2 => '0',
            addr_a_2 => hessian_grad_c_0_ram_2_addr,
            din_a_2 => hessian_grad_c_0_ram_2_d,
            we_a_3 => '0',
            re_a_3 => hessian_output_ram_2_re,
            addr_a_3 => hessian_output_ram_2_addr,
            din_a_3 => (OTHERS => '0'),
            dout_a => ram_2_dout_a,
            clk_b => core_clk,
            sel_b => hessian_ram_2_b_user,
            we_b_0 => '0',
            re_b_0 => hessian_grad_rr_cc_ram_2_b_re,
            addr_b_0 => hessian_grad_rr_cc_ram_2_b_addr,
            din_b_0 => (OTHERS => '0'),
            we_b_1 => '0',
            re_b_1 => '0',
            addr_b_1 => (OTHERS => '0'),
            din_b_1 => (OTHERS => '0'),
            we_b_2 => '0',
            re_b_2 => '0',
            addr_b_2 => (OTHERS => '0'),
            din_b_2 => (OTHERS => '0'),
            we_b_3 => '0',
            re_b_3 => '0',
            addr_b_3 => (OTHERS => '0'),
            din_b_3 => (OTHERS => '0'),
            dout_b => ram_2_dout_b
        );

    bram_hessian_3 : ENTITY work.bram_tdp
        GENERIC MAP(
            DATA_WIDTH => 16,
            DATA_LEN => constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y,
            ADDR_WIDTH => constants.HESSIAN_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => core_clk,
            ce_a => '1',
            we_a => hessian_grad_rr_cc_ram_3_we,
            addr_a => hessian_grad_rr_cc_ram_3_addr,
            din_a => hessian_grad_rr_cc_ram_3_d,
            dout_a => OPEN,
            clk_b => core_clk,
            ce_b => '1',
            we_b => '0',
            addr_b => hessian_output_ram_3_addr,
            din_b => (OTHERS => '0'),
            dout_b => hessian_output_ram_3_d
        );

    -- swap together with vga, but always point to different ram
    hessian_output_ram_swap_trg <= vga_swap_trg_from_core;
    hessian_output_ram_swapper : ENTITY work.bram_swapper_w
        GENERIC MAP(
            INIT_USE_0 => FALSE,
            ADDR_BITS => constants.HESSIAN_OUTPUT_ADDR_BITS,
            DATA_BITS => 2
        )
        PORT MAP(
            clk => core_clk,
            trg => hessian_output_ram_swap_trg,
            we => hessian_output_we,
            addr => hessian_output_addr,
            d => hessian_output_d(7 DOWNTO 6),
            bram_we_0 => hessian_output_ram_we_0,
            bram_addr_0 => hessian_output_ram_addr_0,
            bram_d_0 => hessian_output_ram_d_0,
            bram_we_1 => hessian_output_ram_we_1,
            bram_addr_1 => hessian_output_ram_addr_1,
            bram_d_1 => hessian_output_ram_d_1
        );

    bram_to_vga_0 : ENTITY work.bram_tdp
        GENERIC MAP(
            DATA_WIDTH => 2,
            DATA_LEN => constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y,
            ADDR_WIDTH => constants.HESSIAN_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => core_clk,
            ce_a => '1',
            we_a => hessian_output_ram_we_0,
            addr_a => hessian_output_ram_addr_0,
            din_a => hessian_output_ram_d_0,
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
            DATA_WIDTH => 2,
            DATA_LEN => constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y,
            ADDR_WIDTH => constants.HESSIAN_OUTPUT_ADDR_BITS
        )
        PORT MAP(
            clk_a => core_clk,
            ce_a => '1',
            we_a => hessian_output_ram_we_1,
            addr_a => hessian_output_ram_addr_1,
            din_a => hessian_output_ram_d_1,
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
            bram_d_0 => vga_ram_d_0 & b"000000",
            bram_we_1 => vga_ram_we_1,
            bram_addr_1 => vga_ram_addr_1,
            bram_d_1 => vga_ram_d_1 & b"000000"
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