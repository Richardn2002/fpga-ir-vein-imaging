LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

USE work.constants.HESSIAN_OUTPUT_X;
USE work.constants.HESSIAN_OUTPUT_Y;
USE work.constants.HESSIAN_OUTPUT_ADDR_BITS;

ENTITY hessian_grad_c IS
    PORT (
        clk : IN STD_LOGIC;
        start : IN STD_LOGIC;
        done : OUT STD_LOGIC;
        conv0_addr : OUT STD_LOGIC_VECTOR(HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        conv1_addr : OUT STD_LOGIC_VECTOR(HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        conv0_dout : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
        conv1_dout : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
        gc_addr : OUT STD_LOGIC_VECTOR(HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        gc_din : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
        gc_we : OUT STD_LOGIC
    );
END hessian_grad_c;

ARCHITECTURE Behavioral OF hessian_grad_c IS
    TYPE state_type IS (IDLE, PREPARE, WAIT_READ, COMPUTE, WRITE_OUTPUT, FINISHED);
    SIGNAL state : state_type := IDLE;
    ATTRIBUTE fsm_safe_state : STRING;
    ATTRIBUTE fsm_safe_state OF state : SIGNAL IS "power_on_state";

    SIGNAL y_cnt : INTEGER RANGE 0 TO HESSIAN_OUTPUT_Y := 0;
    SIGNAL x_cnt : INTEGER RANGE 0 TO HESSIAN_OUTPUT_X := 0;
    SIGNAL gc_calc : signed(15 DOWNTO 0);

    SIGNAL conv0_dout_signed : signed(15 DOWNTO 0);
    SIGNAL conv1_dout_signed : signed(15 DOWNTO 0);
    SIGNAL gc_din_signed : signed(15 DOWNTO 0);
BEGIN
    conv0_dout_signed <= signed(conv0_dout);
    conv1_dout_signed <= signed(conv1_dout);
    gc_din <= STD_LOGIC_VECTOR(gc_din_signed);

    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            CASE state IS
                WHEN IDLE =>
                    state <= IDLE;
                    y_cnt <= 0;
                    x_cnt <= 0;
                    done <= '0';
                    gc_we <= '0';
                    IF start = '1' THEN

                        state <= PREPARE;
                    END IF;

                WHEN PREPARE =>
                    gc_we <= '0';
                    IF x_cnt = 0 THEN
                        conv0_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X, HESSIAN_OUTPUT_ADDR_BITS));
                        conv1_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + 1, HESSIAN_OUTPUT_ADDR_BITS));
                    ELSIF x_cnt = HESSIAN_OUTPUT_X - 1 THEN
                        conv0_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt - 1, HESSIAN_OUTPUT_ADDR_BITS));
                        conv1_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt, HESSIAN_OUTPUT_ADDR_BITS));
                    ELSE
                        conv0_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt - 1, HESSIAN_OUTPUT_ADDR_BITS));
                        conv1_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt + 1, HESSIAN_OUTPUT_ADDR_BITS));
                    END IF;
                    state <= WAIT_READ;

                WHEN WAIT_READ =>
                    state <= COMPUTE;

                WHEN COMPUTE =>
                    IF x_cnt = 0 OR x_cnt = HESSIAN_OUTPUT_X - 1 THEN
                        gc_calc <= shift_left(conv1_dout_signed - conv0_dout_signed, 1); -- multiplied by 2 for normalization
                    ELSE
                        gc_calc <= conv1_dout_signed - conv0_dout_signed;
                    END IF;
                    state <= WRITE_OUTPUT;

                WHEN WRITE_OUTPUT =>
                    gc_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt, HESSIAN_OUTPUT_ADDR_BITS));
                    gc_din_signed <= gc_calc;
                    gc_we <= '1';

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
                    gc_we <= '0';
                    done <= '1';
                    IF start = '0' THEN
                        state <= IDLE;
                    END IF;
            END CASE;
        END IF;
    END PROCESS;
END Behavioral;