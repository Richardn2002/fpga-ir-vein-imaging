LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY HESSIAN_conv_c IS
    PORT (
        clk : IN STD_LOGIC;
        trg : IN STD_LOGIC;
        rdy : OUT STD_LOGIC;

        conv_in_en   : OUT STD_LOGIC;
        conv_in_addr : OUT STD_LOGIC_VECTOR(12 DOWNTO 0);
        conv_in_d    : IN  STD_LOGIC_VECTOR(15 DOWNTO 0);

        conv_out_en   : OUT STD_LOGIC;
        conv_out_addr : OUT STD_LOGIC_VECTOR(12 DOWNTO 0);
        conv_out_d    : OUT STD_LOGIC_VECTOR(15 DOWNTO 0)
    );
END HESSIAN_conv_c;

ARCHITECTURE rtl OF HESSIAN_conv_c IS

    CONSTANT IN_X   : INTEGER := 90;
    CONSTANT IN_Y   : INTEGER := 90;
    CONSTANT OUT_X  : INTEGER := 90;
    CONSTANT OUT_Y  : INTEGER := 90;

    TYPE state_t IS (
        IDLE,
        COL_PRIME_REQ,
        COL_PRIME_WAIT1,
        COL_PRIME_WAIT2,
        PIX_REQ,
        PIX_WAIT1,
        WRITE_OUT,
        NEXT_PIXEL,
        DONE
    );

    SIGNAL state, state_next : state_t;

    SIGNAL x, x_next : INTEGER RANGE 0 TO OUT_X - 1;
    SIGNAL y, y_next : INTEGER RANGE 0 TO OUT_Y - 1;

    SIGNAL w0, w1, w2, w3, w4       : unsigned(15 DOWNTO 0);
    SIGNAL w0_next, w1_next, w2_next, w3_next, w4_next : unsigned(15 DOWNTO 0);

    SIGNAL sum_int  : unsigned(15 DOWNTO 0);
    SIGNAL sum_next : unsigned(15 DOWNTO 0);

    SIGNAL prime_cnt, prime_cnt_next : INTEGER RANGE 0 TO 4;

    SIGNAL in_addr_reg, in_addr_next : unsigned(12 DOWNTO 0);

BEGIN


    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            state      <= state_next;
            x          <= x_next;
            y          <= y_next;
            prime_cnt  <= prime_cnt_next;

            w0 <= w0_next;
            w1 <= w1_next;
            w2 <= w2_next;
            w3 <= w3_next;
            w4 <= w4_next;

            sum_int <= sum_next;

            in_addr_reg <= in_addr_next;  
        END IF;
    END PROCESS;


    PROCESS (
        state, trg, x, y, prime_cnt,
        w0, w1, w2, w3, w4,
        sum_int,
        conv_in_d,
        in_addr_reg
    )
        VARIABLE out_addr : INTEGER;
        VARIABLE c        : unsigned(15 DOWNTO 0);

        VARIABLE row_idx_v : INTEGER;           
        VARIABLE addr_v    : unsigned(12 DOWNTO 0); 
    BEGIN

        -- defaults
        state_next      <= state;
        x_next          <= x;
        y_next          <= y;
        prime_cnt_next  <= prime_cnt;

        w0_next <= w0;
        w1_next <= w1;
        w2_next <= w2;
        w3_next <= w3;
        w4_next <= w4;

        sum_next <= sum_int;

        conv_in_en   <= '0';
        conv_in_addr <= (OTHERS => '0');

        conv_out_en   <= '0';
        conv_out_addr <= (OTHERS => '0');
        conv_out_d    <= (OTHERS => '0');

        rdy <= '0';

        in_addr_next <= in_addr_reg;

        CASE state IS

            WHEN IDLE =>
                IF trg = '1' THEN
                    x_next         <= 0;
                    y_next         <= 0;
                    prime_cnt_next <= 0;

                    w0_next <= (OTHERS => '0');
                    w1_next <= (OTHERS => '0');
                    w2_next <= (OTHERS => '0');
                    w3_next <= (OTHERS => '0');
                    w4_next <= (OTHERS => '0');
                    sum_next <= (OTHERS => '0');

                    in_addr_next <= (OTHERS => '0');

                    state_next <= COL_PRIME_REQ;
                END IF;

            WHEN COL_PRIME_REQ =>


                CASE prime_cnt IS
                    WHEN 0 | 1 | 2 => row_idx_v := 0;
                    WHEN 3         => row_idx_v := 1;
                    WHEN 4         => row_idx_v := 2;
                    WHEN OTHERS    => row_idx_v := 0;
                END CASE;

                addr_v := to_unsigned(row_idx_v * IN_X + x, 13);

                conv_in_en   <= '1';
                conv_in_addr <= std_logic_vector(addr_v);
                in_addr_next <= addr_v;

                state_next <= COL_PRIME_WAIT1;

            WHEN COL_PRIME_WAIT1 =>
                conv_in_en   <= '1';
                conv_in_addr <= std_logic_vector(in_addr_reg);
                state_next   <= COL_PRIME_WAIT2;

            WHEN COL_PRIME_WAIT2 =>
                c := unsigned(conv_in_d);

                CASE prime_cnt IS
                    WHEN 0 => w0_next <= c;
                    WHEN 1 => w1_next <= c;
                    WHEN 2 => w2_next <= c;
                    WHEN 3 => w3_next <= c;
                    WHEN 4 => w4_next <= c;
                    WHEN OTHERS => NULL;
                END CASE;

                IF prime_cnt = 4 THEN
                    state_next <= PIX_REQ;
                ELSE
                    prime_cnt_next <= prime_cnt + 1;
                    state_next     <= COL_PRIME_REQ;
                END IF;

            WHEN PIX_REQ =>
                IF (y + 3) < IN_Y THEN
                    row_idx_v := y + 3;
                ELSE
                    row_idx_v := IN_Y - 1;
                END IF;

                addr_v := to_unsigned(row_idx_v * IN_X + x, 13);

                conv_in_en   <= '1';
                conv_in_addr <= std_logic_vector(addr_v);
                in_addr_next <= addr_v;

                state_next <= PIX_WAIT1;

            WHEN PIX_WAIT1 =>
                conv_in_en   <= '1';
                conv_in_addr <= std_logic_vector(in_addr_reg);

                sum_next <= w0 + w1 + w2 + w3 + w4;

                c := unsigned(conv_in_d);

                w0_next <= w1;
                w1_next <= w2;
                w2_next <= w3;
                w3_next <= w4;
                w4_next <= c;

                state_next <= WRITE_OUT;

            WHEN WRITE_OUT =>
                out_addr := y * OUT_X + x;

                conv_out_en   <= '1';
                conv_out_addr <= std_logic_vector(to_unsigned(out_addr, 13));
                conv_out_d    <= std_logic_vector(sum_int);

                state_next    <= NEXT_PIXEL;

            WHEN NEXT_PIXEL =>
                IF y = OUT_Y - 1 THEN
                    y_next <= 0;

                    IF x = OUT_X - 1 THEN
                        state_next <= DONE;
                    ELSE
                        x_next         <= x + 1;
                        prime_cnt_next <= 0;
                        state_next     <= COL_PRIME_REQ;
                    END IF;
                ELSE
                    y_next     <= y + 1;
                    state_next <= PIX_REQ;
                END IF;

            WHEN DONE =>
                rdy <= '1';
                IF trg = '0' THEN
                    state_next <= IDLE;
                END IF;

        END CASE;
    END PROCESS;

END rtl;
