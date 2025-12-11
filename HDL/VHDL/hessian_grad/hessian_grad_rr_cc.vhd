-- Module 3: hessian_grad_rr_cc
LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE work.hessian_constants.ALL;

ENTITY hessian_grad_rr_cc IS
    PORT (
        clk : IN STD_LOGIC;
        rst : IN STD_LOGIC;
        start : IN STD_LOGIC;
        done : OUT STD_LOGIC;
        gr0_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        gr0_dout : IN signed(15 DOWNTO 0);
        gr1_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        gr1_dout : IN signed(15 DOWNTO 0);
        gc0_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        gc0_dout : IN signed(15 DOWNTO 0);
        gc1_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        gc1_dout : IN signed(15 DOWNTO 0);
        rr_p_cc_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        rr_p_cc_din : OUT signed(15 DOWNTO 0);
        rr_p_cc_we : OUT STD_LOGIC;
        rr_m_cc_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        rr_m_cc_din : OUT signed(15 DOWNTO 0);
        rr_m_cc_we : OUT STD_LOGIC
    );
END hessian_grad_rr_cc;

ARCHITECTURE Behavioral OF hessian_grad_rr_cc IS
    TYPE state_type IS (IDLE, PREPARE, WAIT_READ, COMPUTE, WRITE_OUTPUT, FINISHED);
    SIGNAL state : state_type := IDLE;
    SIGNAL y_cnt : INTEGER RANGE 0 TO HESSIAN_OUTPUT_Y := 0;
    SIGNAL x_cnt : INTEGER RANGE 0 TO HESSIAN_OUTPUT_X := 0;
    SIGNAL rr_p_cc_term_out, rr_m_cc_term_out : signed(15 DOWNTO 0);
BEGIN
    PROCESS (clk)
        VARIABLE hrr, hcc : signed(15 DOWNTO 0);
        VARIABLE hrr_minus_hcc : signed(15 DOWNTO 0);
        VARIABLE diff_squared : signed(31 DOWNTO 0);
    BEGIN
        IF rising_edge(clk) THEN
            IF rst = '1' THEN
                state <= IDLE;
                y_cnt <= 0;
                x_cnt <= 0;
                done <= '0';
                rr_p_cc_we <= '0';
                rr_m_cc_we <= '0';
            ELSE
                CASE state IS
                    WHEN IDLE =>
                        done <= '0';
                        rr_p_cc_we <= '0';
                        rr_m_cc_we <= '0';
                        IF start = '1' THEN
                            y_cnt <= 0;
                            x_cnt <= 0;
                            state <= PREPARE;
                        END IF;

                    WHEN PREPARE =>
                        rr_p_cc_we <= '0';
                        rr_m_cc_we <= '0';
                        IF y_cnt = 0 THEN
                            gr0_addr <= STD_LOGIC_VECTOR(to_unsigned(x_cnt, 14));
                            gr1_addr <= STD_LOGIC_VECTOR(to_unsigned(HESSIAN_OUTPUT_X + x_cnt, 14));
                        ELSIF y_cnt = HESSIAN_OUTPUT_Y - 1 THEN
                            gr0_addr <= STD_LOGIC_VECTOR(to_unsigned((y_cnt - 1) * HESSIAN_OUTPUT_X + x_cnt, 14));
                            gr1_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt, 14));
                        ELSE
                            gr0_addr <= STD_LOGIC_VECTOR(to_unsigned((y_cnt - 1) * HESSIAN_OUTPUT_X + x_cnt, 14));
                            gr1_addr <= STD_LOGIC_VECTOR(to_unsigned((y_cnt + 1) * HESSIAN_OUTPUT_X + x_cnt, 14));
                        END IF;
                        IF x_cnt = 0 THEN
                            gc0_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X, 14));
                            gc1_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + 1, 14));
                        ELSIF x_cnt = HESSIAN_OUTPUT_X - 1 THEN
                            gc0_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt - 1, 14));
                            gc1_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt, 14));
                        ELSE
                            gc0_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt - 1, 14));
                            gc1_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt + 1, 14));
                        END IF;
                        state <= WAIT_READ;

                    WHEN WAIT_READ =>
                        state <= COMPUTE;

                    WHEN COMPUTE =>
                        -- Hrr
                        IF y_cnt = 0 OR y_cnt = HESSIAN_OUTPUT_Y - 1 THEN
                            hrr := shift_left(gr1_dout - gr0_dout, 1);
                        ELSE
                            hrr := gr1_dout - gr0_dout;
                        END IF;

                        -- Hcc
                        IF x_cnt = 0 OR x_cnt = HESSIAN_OUTPUT_X - 1 THEN
                            hcc := shift_left(gc1_dout - gc0_dout, 1);
                        ELSE
                            hcc := gc1_dout - gc0_dout;
                        END IF;

                        -- Output
                        rr_p_cc_term_out <= shift_right(hrr + hcc, 3);

                        hrr_minus_hcc := hrr - hcc;
                        diff_squared := hrr_minus_hcc * hrr_minus_hcc;
                        rr_m_cc_term_out <= resize(shift_right(diff_squared, 4), 16);

                        state <= WRITE_OUTPUT;

                    WHEN WRITE_OUTPUT =>
                        rr_p_cc_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt, 14));
                        rr_p_cc_din <= rr_p_cc_term_out;
                        rr_p_cc_we <= '1';

                        rr_m_cc_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt, 14));
                        rr_m_cc_din <= rr_m_cc_term_out;
                        rr_m_cc_we <= '1';

                        IF x_cnt = HESSIAN_OUTPUT_X - 1 THEN
                            x_cnt <= 0;
                            IF y_cnt = HESSIAN_OUTPUT_Y - 1 THEN
                                state <= FINISHED;
                            ELSE
                                y_cnt <= y_cnt + 1;
                                state <= PREPARE;
                            END IF;
                        ELSE
                            x_cnt <= x_cnt + 1;
                            state <= PREPARE;
                        END IF;

                    WHEN FINISHED =>
                        rr_p_cc_we <= '0';
                        rr_m_cc_we <= '0';
                        done <= '1';
                        IF start = '0' THEN
                            state <= IDLE;
                        END IF;
                END CASE;
            END IF;
        END IF;
    END PROCESS;
END Behavioral;