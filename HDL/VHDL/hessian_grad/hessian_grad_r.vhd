LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE work.hessian_constants.ALL;

ENTITY hessian_grad_r IS
    PORT (
        clk : IN STD_LOGIC;
        rst : IN STD_LOGIC;
        start : IN STD_LOGIC;
        done : OUT STD_LOGIC;
        conv0_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        conv1_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        conv0_dout : IN signed(15 DOWNTO 0);
        conv1_dout : IN signed(15 DOWNTO 0);
        gr_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        gr_din : OUT signed(15 DOWNTO 0);
        gr_we : OUT STD_LOGIC
    );
END hessian_grad_r;

ARCHITECTURE Behavioral OF hessian_grad_r IS
    TYPE state_type IS (IDLE, PREPARE, WAIT_READ, COMPUTE, WRITE_OUTPUT, FINISHED);
    SIGNAL state : state_type := IDLE;
    SIGNAL y_cnt : INTEGER RANGE 0 TO HESSIAN_OUTPUT_Y := 0;
    SIGNAL x_cnt : INTEGER RANGE 0 TO HESSIAN_OUTPUT_X := 0;
    SIGNAL gr_calc : signed(15 DOWNTO 0);

BEGIN
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF rst = '1' THEN
                state <= IDLE;
                y_cnt <= 0;
                x_cnt <= 0;
                done <= '0';
                gr_we <= '0';
            ELSE
                CASE state IS
                    WHEN IDLE =>
                        done <= '0';
                        gr_we <= '0';
                        IF start = '1' THEN
                            y_cnt <= 0;
                            x_cnt <= 0;
                            state <= PREPARE;
                        END IF;

                    WHEN PREPARE =>
                        gr_we <= '0'; --loopback from WRITE_OUT, so need to set WE back to 0
                        IF y_cnt = 0 THEN -- top boundary, top cell addr and addr of the cell below it
                            conv0_addr <= STD_LOGIC_VECTOR(to_unsigned(x_cnt, 14));
                            conv1_addr <= STD_LOGIC_VECTOR(to_unsigned(HESSIAN_OUTPUT_X + x_cnt, 14)); -- prepares difference
                        ELSIF y_cnt = HESSIAN_OUTPUT_Y - 1 THEN -- bottom boundary
                            conv0_addr <= STD_LOGIC_VECTOR(to_unsigned((y_cnt - 1) * HESSIAN_OUTPUT_X + x_cnt, 14));
                            conv1_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt, 14));
                        ELSE
                            conv0_addr <= STD_LOGIC_VECTOR(to_unsigned((y_cnt - 1) * HESSIAN_OUTPUT_X + x_cnt, 14));
                            conv1_addr <= STD_LOGIC_VECTOR(to_unsigned((y_cnt + 1) * HESSIAN_OUTPUT_X + x_cnt, 14));
                        END IF;
                        state <= WAIT_READ;

                    WHEN WAIT_READ =>
                        -- RAM will update 'conv0_dout' 
                        state <= COMPUTE;

                    WHEN COMPUTE =>
                        IF y_cnt = 0 OR y_cnt = HESSIAN_OUTPUT_Y - 1 THEN
                            gr_calc <= shift_left(conv1_dout - conv0_dout, 1);
                        ELSE
                            gr_calc <= conv1_dout - conv0_dout;
                        END IF;
                        state <= WRITE_OUTPUT;

                    WHEN WRITE_OUTPUT =>
                        gr_addr <= STD_LOGIC_VECTOR(to_unsigned(y_cnt * HESSIAN_OUTPUT_X + x_cnt, 14));
                        gr_din <= gr_calc;
                        gr_we <= '1';

                        IF x_cnt = HESSIAN_OUTPUT_X - 1 THEN -- check if at end of the row
                            x_cnt <= 0;
                            IF y_cnt = HESSIAN_OUTPUT_Y - 1 THEN -- last row
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
                        gr_we <= '0';
                        done <= '1';
                        IF start = '0' THEN
                            state <= IDLE;
                        END IF;
                END CASE;
            END IF;
        END IF;
    END PROCESS;
END Behavioral;