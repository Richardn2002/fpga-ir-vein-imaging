LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;

USE work.constants;

ENTITY hessian_output IS
    PORT (
        clk : IN STD_LOGIC;

        trg : IN STD_LOGIC;
        rdy : OUT STD_LOGIC;

        -- Input RAM ports (address out, data in)
        rrp_addr : OUT STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        rrp_dout : IN STD_LOGIC_VECTOR(15 DOWNTO 0);

        rrm_addr : OUT STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        rrm_dout : IN STD_LOGIC_VECTOR(15 DOWNTO 0);

        rc_addr : OUT STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        rc_dout : IN STD_LOGIC_VECTOR(15 DOWNTO 0);

        out_addr : OUT STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        out_din : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
        out_we : OUT STD_LOGIC
    );
END ENTITY;

ARCHITECTURE rtl OF hessian_output IS
    CONSTANT N : INTEGER := constants.HESSIAN_OUTPUT_X * constants.HESSIAN_OUTPUT_Y;

    TYPE state_t IS (IDLE, ISSUE_RD, WAIT_RD, COMPUTE, WRITE, DONE);
    SIGNAL state : state_t := IDLE;
    ATTRIBUTE fsm_safe_state : STRING;
    ATTRIBUTE fsm_safe_state OF state : SIGNAL IS "power_on_state";

    -- address / pixel index
    SIGNAL idx : INTEGER RANGE 0 TO N - 1 := 0;
    SIGNAL addr_u : unsigned(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0) := (OTHERS => '0');

    -- latched input data for this pixel
    SIGNAL rr_p_cc_r : signed(15 DOWNTO 0) := (OTHERS => '0');
    SIGNAL rr_m_cc_r : signed(15 DOWNTO 0) := (OTHERS => '0');
    SIGNAL rc_r : signed(15 DOWNTO 0) := (OTHERS => '0');

    TYPE sqrt_bin_t IS ARRAY (0 TO 15) OF INTEGER RANGE 0 TO 65535;
    CONSTANT SQRT_BINS : sqrt_bin_t := (
        0, 2, 6, 12, 20, 30, 42, 56,
        72, 90, 110, 132, 156, 182, 210, 240
    );

    PURE
    FUNCTION sqrt_quant(v : INTEGER) RETURN INTEGER IS
        VARIABLE c00 : STD_LOGIC := '0';
        VARIABLE c01 : STD_LOGIC := '0';
        VARIABLE c02 : STD_LOGIC := '0';
        VARIABLE c03 : STD_LOGIC := '0';
        VARIABLE c04 : STD_LOGIC := '0';
        VARIABLE c05 : STD_LOGIC := '0';
        VARIABLE c06 : STD_LOGIC := '0';
        VARIABLE c07 : STD_LOGIC := '0';
        VARIABLE c08 : STD_LOGIC := '0';
        VARIABLE c09 : STD_LOGIC := '0';
        VARIABLE c10 : STD_LOGIC := '0';
        VARIABLE c11 : STD_LOGIC := '0';
        VARIABLE c12 : STD_LOGIC := '0';
        VARIABLE c13 : STD_LOGIC := '0';
        VARIABLE c14 : STD_LOGIC := '0';
        VARIABLE c15 : STD_LOGIC := '0';
    BEGIN
        IF v > SQRT_BINS(00) THEN
            c00 := '1';
        END IF;
        IF v > SQRT_BINS(01) THEN
            c01 := '1';
        END IF;
        IF v > SQRT_BINS(02) THEN
            c02 := '1';
        END IF;
        IF v > SQRT_BINS(03) THEN
            c03 := '1';
        END IF;
        IF v > SQRT_BINS(04) THEN
            c04 := '1';
        END IF;
        IF v > SQRT_BINS(05) THEN
            c05 := '1';
        END IF;
        IF v > SQRT_BINS(06) THEN
            c06 := '1';
        END IF;
        IF v > SQRT_BINS(07) THEN
            c07 := '1';
        END IF;
        IF v > SQRT_BINS(08) THEN
            c08 := '1';
        END IF;
        IF v > SQRT_BINS(09) THEN
            c09 := '1';
        END IF;
        IF v > SQRT_BINS(10) THEN
            c10 := '1';
        END IF;
        IF v > SQRT_BINS(11) THEN
            c11 := '1';
        END IF;
        IF v > SQRT_BINS(12) THEN
            c12 := '1';
        END IF;
        IF v > SQRT_BINS(13) THEN
            c13 := '1';
        END IF;
        IF v > SQRT_BINS(14) THEN
            c14 := '1';
        END IF;
        IF v > SQRT_BINS(15) THEN
            c15 := '1';
        END IF;

        CASE STD_LOGIC_VECTOR'(c00 & c01 & c02 & c03 & c04 & c05 & c06 & c07 & c08 & c09 & c10 & c11 & c12 & c13 & c14 & c15) IS
            WHEN b"0000000000000000" => RETURN 00;
            WHEN b"1000000000000000" => RETURN 01;
            WHEN b"1100000000000000" => RETURN 02;
            WHEN b"1110000000000000" => RETURN 03;
            WHEN b"1111000000000000" => RETURN 04;
            WHEN b"1111100000000000" => RETURN 05;
            WHEN b"1111110000000000" => RETURN 06;
            WHEN b"1111111000000000" => RETURN 07;
            WHEN b"1111111100000000" => RETURN 08;
            WHEN b"1111111110000000" => RETURN 09;
            WHEN b"1111111111000000" => RETURN 10;
            WHEN b"1111111111100000" => RETURN 11;
            WHEN b"1111111111110000" => RETURN 12;
            WHEN b"1111111111111000" => RETURN 13;
            WHEN b"1111111111111100" => RETURN 14;
            WHEN b"1111111111111110" => RETURN 15;
            WHEN b"1111111111111111" => RETURN 16;
            WHEN OTHERS => RETURN 00;
        END CASE;
    END FUNCTION;

    SIGNAL final_out_reg : INTEGER RANGE 0 TO 65535 := 0;
BEGIN
    -- Drive addresses continuously from addr_u (set in FSM)
    rrp_addr <= STD_LOGIC_VECTOR(addr_u);
    rrm_addr <= STD_LOGIC_VECTOR(addr_u);
    rc_addr <= STD_LOGIC_VECTOR(addr_u);

    PROCESS (clk)
        VARIABLE rc_i : INTEGER;
        VARIABLE rc_sq_full : INTEGER;
        VARIABLE rc_sq_div4 : INTEGER;

        VARIABLE sqrt_in_full : INTEGER;
        VARIABLE sqrt_in_s16 : signed(15 DOWNTO 0);
        VARIABLE sqrt_in_wrap : INTEGER;

        VARIABLE sqrt_val_v : INTEGER;
        VARIABLE tmp_sum_v : INTEGER;
        VARIABLE final_out_v : INTEGER;
    BEGIN
        IF rising_edge(clk) THEN
            CASE state IS
                WHEN IDLE =>
                    state <= IDLE;
                    rdy <= '0';
                    out_we <= '0';

                    idx <= 0;
                    addr_u <= (OTHERS => '0');
                    out_addr <= (OTHERS => '0');

                    rr_p_cc_r <= (OTHERS => '0');
                    rr_m_cc_r <= (OTHERS => '0');
                    rc_r <= (OTHERS => '0');

                    out_din <= (OTHERS => '0');
                    final_out_reg <= 0;

                    IF trg = '1' THEN
                        state <= ISSUE_RD;
                    END IF;

                WHEN ISSUE_RD =>
                    out_we <= '0';
                    state <= WAIT_RD;

                WHEN WAIT_RD =>
                    out_we <= '0';

                    rr_p_cc_r <= SIGNED(rrp_dout);
                    rr_m_cc_r <= SIGNED(rrm_dout);
                    rc_r <= SIGNED(rc_dout);
                    state <= COMPUTE;

                WHEN COMPUTE =>
                    -- rc^2//4
                    rc_i := to_integer(rc_r);
                    rc_sq_full := rc_i * rc_i;
                    rc_sq_div4 := rc_sq_full / 4;

                    -- sqrt_in_full
                    sqrt_in_full := rc_sq_div4 + to_integer(rr_m_cc_r) + 1;

                    -- wrap to int16 before quant, to match numpy behavior
                    sqrt_in_s16 := to_signed(sqrt_in_full, 16);
                    sqrt_in_wrap := to_integer(sqrt_in_s16);

                    sqrt_val_v := sqrt_quant(sqrt_in_wrap);

                    tmp_sum_v := (to_integer(rr_p_cc_r) + sqrt_val_v) * 64;
                    final_out_v := tmp_sum_v / 16;

                    IF final_out_v < 0 THEN
                        final_out_v := - final_out_v;
                    END IF;

                    final_out_reg <= final_out_v;
                    state <= WRITE;

                WHEN WRITE =>
                    out_we <= '1';
                    out_din <= STD_LOGIC_VECTOR(to_signed(final_out_reg, 16));
                    out_addr <= STD_LOGIC_VECTOR(addr_u);

                    IF idx = N - 1 THEN
                        state <= DONE;
                    ELSE
                        idx <= idx + 1;
                        addr_u <= to_unsigned(idx + 1, constants.HESSIAN_OUTPUT_ADDR_BITS);
                        state <= ISSUE_RD;
                    END IF;

                WHEN DONE =>
                    out_we <= '0';
                    rdy <= '1';
                    IF trg = '0' THEN
                        state <= IDLE;
                    END IF;

            END CASE;
        END IF;
    END PROCESS;
END ARCHITECTURE;