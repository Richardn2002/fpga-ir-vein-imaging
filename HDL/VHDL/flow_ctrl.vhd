LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY flow_ctrl IS
    PORT (
        core_clk : IN STD_LOGIC;

        cam_frame_writing : IN STD_LOGIC;
        cam_ram_swap_trg : OUT STD_LOGIC;

        clahe_mapping_trg : OUT STD_LOGIC;
        clahe_mapping_rdy : IN STD_LOGIC;
        clahe_output_trg : OUT STD_LOGIC;
        clahe_output_rdy : IN STD_LOGIC;
        hessian_conv_r_trg : OUT STD_LOGIC;
        hessian_conv_r_rdy : IN STD_LOGIC;
        hessian_conv_c_trg : OUT STD_LOGIC;
        hessian_conv_c_rdy : IN STD_LOGIC;
        hessian_grad_r_trg : OUT STD_LOGIC;
        hessian_grad_r_rdy : IN STD_LOGIC;
        hessian_grad_c_0_trg : OUT STD_LOGIC;
        hessian_grad_c_0_rdy : IN STD_LOGIC;
        hessian_grad_rr_cc_trg : OUT STD_LOGIC;
        hessian_grad_rr_cc_rdy : IN STD_LOGIC;
        hessian_grad_c_1_trg : OUT STD_LOGIC;
        hessian_grad_c_1_rdy : IN STD_LOGIC;
        hessian_output_trg : OUT STD_LOGIC;
        hessian_output_rdy : IN STD_LOGIC;

        clahe_reader_swap_trg : OUT STD_LOGIC;
        hessian_ram_0_a_user : OUT NATURAL RANGE 0 TO 3;
        hessian_ram_0_b_user : OUT NATURAL RANGE 0 TO 3;
        hessian_ram_1_a_user : OUT NATURAL RANGE 0 TO 3;
        hessian_ram_1_b_user : OUT NATURAL RANGE 0 TO 3;
        hessian_ram_2_a_user : OUT NATURAL RANGE 0 TO 3;
        hessian_ram_2_b_user : OUT NATURAL RANGE 0 TO 3;

        vga_okay_to_swap : IN STD_LOGIC;
        vga_ram_swap_trg : OUT STD_LOGIC
    );
END ENTITY;

-- HESSIAN_RAM_0:
-- A0, R, BY HESSIAN_CONV_C
-- A1, R, BY HESSIAN_GRAD_C_1
-- A2, R, BY HESSIAN_GRAD_RR_CC
-- A3, X
-- B0, W, BY HESSIAN_CONV_R
-- B1, W, BY HESSIAN_GRAD_R
-- B2, R, BY HESSIAN_GRAD_C_1
-- B3, R, BY HESSIAN_GRAD_RR_CC
-- 
-- HESSIAN_RAM_1:
-- A0, R, BY HESSIAN_GRAD_R
-- A1, R, BY HESSIAN_GRAD_C_0
-- A2, R, BY HESSIAN_OUTPUT
-- A3, X
-- B0, W, BY HESSIAN_CONV_C
-- B1, R, BY HESSIAN_GRAD_R
-- B2, W, BY HESSIAN_GRAD_RR_CC
-- B3, R, BY HESSIAN_GRAD_C_0
-- 
-- HESSIAN_RAM_2:
-- A0, W, BY HESSIAN_GRAD_C_1,
-- A1, R, BY HESSIAN_GRAD_RR_CC
-- A2, W, BY HESSIAN_GRAD_C_0,
-- A3, R, BY HESSIAN_OUTPUT
-- B0, R, BY HESSIAN_GRAD_RR_CC
-- B1, X
-- B2, X
-- B3, X
-- 
-- HESSIAN_RAM_3:
-- A, W, BY HESSIAN_GRAD_RR_CC
-- B, R, BY HESSIAN_OUTPUT

ARCHITECTURE arch OF flow_ctrl IS
    SIGNAL new_frame_incoming : BOOLEAN := FALSE;
    SIGNAL cam_writing_frame_last : STD_LOGIC := '0';

    TYPE s_type IS (
        WAIT_FOR_FRAME,
        CLAHE_MAPPING,
        CLAHE_MAPPING_WAIT,
        CLAHE_OUTPUT,
        CLAHE_OUTPUT_WAIT,
        HESSIAN_CONV_R,
        HESSIAN_CONV_R_WAIT,
        HESSIAN_CONV_C,
        HESSIAN_CONV_C_WAIT,
        HESSIAN_GRAD_R,
        HESSIAN_GRAD_R_WAIT,
        HESSIAN_GRAD_C_0,
        HESSIAN_GRAD_C_0_WAIT,
        HESSIAN_GRAD_RR_CC,
        HESSIAN_GRAD_RR_CC_WAIT,
        HESSIAN_GRAD_C_1,
        HESSIAN_GRAD_C_1_WAIT,
        HESSIAN_OUTPUT,
        HESSIAN_OUTPUT_WAIT,
        WAIT_FOR_VGA_FREE
    );
    SIGNAL s : s_type := WAIT_FOR_FRAME;
    SIGNAL s_next : s_type;

    SIGNAL cam_ram_swap_trg_next : STD_LOGIC;
    SIGNAL vga_ram_swap_trg_next : STD_LOGIC;
BEGIN
    PROCESS (core_clk) BEGIN
        IF rising_edge(core_clk) THEN
            cam_writing_frame_last <= cam_frame_writing;
            IF cam_writing_frame_last = '0' AND cam_frame_writing = '1' THEN
                new_frame_incoming <= TRUE;
            END IF;
            IF s = CLAHE_MAPPING THEN
                new_frame_incoming <= FALSE;
            END IF;
        END IF;
    END PROCESS;

    PROCESS (
        s, new_frame_incoming, cam_frame_writing,
        clahe_mapping_rdy,
        clahe_output_rdy,
        hessian_conv_r_rdy,
        hessian_conv_c_rdy,
        hessian_grad_r_rdy,
        hessian_grad_c_0_rdy,
        hessian_grad_rr_cc_rdy,
        hessian_grad_c_1_rdy,
        hessian_output_rdy,
        vga_okay_to_swap
        ) BEGIN
        s_next <= s;
        cam_ram_swap_trg_next <= '0';

        clahe_mapping_trg <= '0';
        clahe_output_trg <= '0';
        hessian_conv_r_trg <= '0';
        hessian_conv_c_trg <= '0';
        hessian_grad_r_trg <= '0';
        hessian_grad_c_0_trg <= '0';
        hessian_grad_rr_cc_trg <= '0';
        hessian_grad_c_1_trg <= '0';
        hessian_output_trg <= '0';
        clahe_reader_swap_trg <= '0';
        hessian_ram_0_a_user <= 0;
        hessian_ram_0_b_user <= 0;
        hessian_ram_1_a_user <= 0;
        hessian_ram_1_b_user <= 0;
        hessian_ram_2_a_user <= 0;
        hessian_ram_2_b_user <= 0;

        vga_ram_swap_trg_next <= '0';

        CASE s IS
            WHEN WAIT_FOR_FRAME =>
                IF new_frame_incoming AND cam_frame_writing = '0' THEN
                    s_next <= CLAHE_MAPPING;
                    cam_ram_swap_trg_next <= '1';
                END IF;

            WHEN CLAHE_MAPPING =>
                clahe_mapping_trg <= '1';
                clahe_reader_swap_trg <= '1';
                s_next <= CLAHE_MAPPING_WAIT;
            WHEN CLAHE_MAPPING_WAIT =>
                IF clahe_mapping_rdy = '1' THEN
                    s_next <= CLAHE_OUTPUT;
                END IF;

            WHEN CLAHE_OUTPUT =>
                clahe_output_trg <= '1';
                clahe_reader_swap_trg <= '1';
                s_next <= CLAHE_OUTPUT_WAIT;
            WHEN CLAHE_OUTPUT_WAIT =>
                IF clahe_output_rdy = '1' THEN
                    s_next <= HESSIAN_CONV_R;
                END IF;

            WHEN HESSIAN_CONV_R =>
                hessian_conv_r_trg <= '1';
                s_next <= HESSIAN_CONV_R_WAIT;
            WHEN HESSIAN_CONV_R_WAIT =>
                hessian_ram_0_b_user <= 0;
                IF hessian_conv_r_rdy = '1' THEN
                    s_next <= HESSIAN_CONV_C;
                END IF;

            WHEN HESSIAN_CONV_C =>
                hessian_conv_c_trg <= '1';
                s_next <= HESSIAN_CONV_C_WAIT;
            WHEN HESSIAN_CONV_C_WAIT =>
                hessian_ram_0_a_user <= 0;
                hessian_ram_1_b_user <= 0;
                IF hessian_conv_c_rdy = '1' THEN
                    s_next <= HESSIAN_GRAD_R;
                END IF;

            WHEN HESSIAN_GRAD_R =>
                hessian_grad_r_trg <= '1';
                s_next <= HESSIAN_GRAD_R_WAIT;
            WHEN HESSIAN_GRAD_R_WAIT =>
                hessian_ram_0_b_user <= 1;
                hessian_ram_1_a_user <= 0;
                hessian_ram_1_b_user <= 1;
                IF hessian_grad_r_rdy = '1' THEN
                    s_next <= HESSIAN_GRAD_C_0;
                END IF;

            WHEN HESSIAN_GRAD_C_0 =>
                hessian_grad_c_0_trg <= '1';
                s_next <= HESSIAN_GRAD_C_0_WAIT;
            WHEN HESSIAN_GRAD_C_0_WAIT =>
                hessian_ram_1_a_user <= 1;
                hessian_ram_1_b_user <= 3;
                hessian_ram_2_a_user <= 2;
                IF hessian_grad_c_0_rdy = '1' THEN
                    s_next <= HESSIAN_GRAD_RR_CC;
                END IF;

            WHEN HESSIAN_GRAD_RR_CC =>
                hessian_grad_rr_cc_trg <= '1';
                s_next <= HESSIAN_GRAD_RR_CC_WAIT;
            WHEN HESSIAN_GRAD_RR_CC_WAIT =>
                hessian_ram_0_a_user <= 2;
                hessian_ram_0_b_user <= 3;
                hessian_ram_1_b_user <= 2;
                hessian_ram_2_a_user <= 1;
                hessian_ram_2_b_user <= 0;
                IF hessian_grad_rr_cc_rdy = '1' THEN
                    s_next <= HESSIAN_GRAD_C_1;
                END IF;

            WHEN HESSIAN_GRAD_C_1 =>
                hessian_grad_c_1_trg <= '1';
                s_next <= HESSIAN_GRAD_C_1_WAIT;
            WHEN HESSIAN_GRAD_C_1_WAIT =>
                hessian_ram_0_a_user <= 1;
                hessian_ram_0_b_user <= 2;
                hessian_ram_2_a_user <= 0;
                IF hessian_grad_c_1_rdy <= '1' THEN
                    s_next <= HESSIAN_OUTPUT;
                END IF;

            WHEN HESSIAN_OUTPUT =>
                hessian_output_trg <= '1';
                s_next <= HESSIAN_OUTPUT_WAIT;
            WHEN HESSIAN_OUTPUT_WAIT =>
                hessian_ram_1_a_user <= 2;
                hessian_ram_2_a_user <= 3;
                IF hessian_output_rdy = '1' THEN
                    s_next <= WAIT_FOR_VGA_FREE;
                END IF;

            WHEN WAIT_FOR_VGA_FREE =>
                IF vga_okay_to_swap = '1' THEN
                    s_next <= WAIT_FOR_FRAME;
                    vga_ram_swap_trg_next <= '1';
                END IF;
        END CASE;
    END PROCESS;

    PROCESS (core_clk) BEGIN
        IF rising_edge(core_clk) THEN
            s <= s_next;
            cam_ram_swap_trg <= cam_ram_swap_trg_next;
            vga_ram_swap_trg <= vga_ram_swap_trg_next;
        END IF;
    END PROCESS;
END ARCHITECTURE;