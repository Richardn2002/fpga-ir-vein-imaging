LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY HESSIAN_conv_r IS
    PORT (
        clk : IN STD_LOGIC;
        trg : IN STD_LOGIC;
        rdy : OUT STD_LOGIC;

        clahe_in_en   : OUT STD_LOGIC;
        clahe_in_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        clahe_in_d    : IN  STD_LOGIC_VECTOR(7 DOWNTO 0);

        conv_out_en   : OUT STD_LOGIC;
        conv_out_addr : OUT STD_LOGIC_VECTOR(12 DOWNTO 0);
        conv_out_d    : OUT STD_LOGIC_VECTOR(15 DOWNTO 0)
    );
END HESSIAN_conv_r;

ARCHITECTURE rtl OF HESSIAN_conv_r IS

    CONSTANT IN_X    : INTEGER := 96;
    CONSTANT OUT_X   : INTEGER := 90;
    CONSTANT OUT_Y   : INTEGER := 90;
    CONSTANT RADIUS  : INTEGER := 3;

    TYPE state_t IS (
        IDLE,
        ROW_PRIME_REQ,
        ROW_PRIME_WAIT1,
        ROW_PRIME_WAIT2,

        PIX_REQ,
        PIX_WAIT1,
        PIX_WAIT2,

        WRITE_OUT,
        NEXT_PIXEL,
        DONE
    );

    SIGNAL state, state_next : state_t;

    SIGNAL x, x_next : INTEGER RANGE 0 TO OUT_X - 1;  -- 0..89
    SIGNAL y, y_next : INTEGER RANGE 0 TO OUT_Y - 1;  -- 0..89

    SIGNAL w0, w1, w2, w3, w4       : unsigned(7 DOWNTO 0);
    SIGNAL w0_next, w1_next, w2_next, w3_next, w4_next : unsigned(7 DOWNTO 0);

    SIGNAL sum_int  : unsigned(15 DOWNTO 0);
    SIGNAL sum_next : unsigned(15 DOWNTO 0);

    SIGNAL prime_cnt, prime_cnt_next : INTEGER RANGE 0 TO 4;

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
        END IF;
    END PROCESS;

    PROCESS (
        state, trg, x, y, prime_cnt,
        w0, w1, w2, w3, w4,
        sum_int,
        clahe_in_d
    )
        VARIABLE in_addr  : INTEGER;
        VARIABLE out_addr : INTEGER;
        VARIABLE c        : unsigned(7 DOWNTO 0);
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

        clahe_in_en   <= '0';
        clahe_in_addr <= (OTHERS => '0');

        conv_out_en   <= '0';
        conv_out_addr <= (OTHERS => '0');
        conv_out_d    <= (OTHERS => '0');

        rdy <= '0';

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

                    state_next <= ROW_PRIME_REQ;
                END IF;

            WHEN ROW_PRIME_REQ =>

                in_addr := (y + RADIUS) * IN_X + (prime_cnt + 1);

                clahe_in_en   <= '1';
                clahe_in_addr <= STD_LOGIC_VECTOR(to_unsigned(in_addr, 14));

                state_next <= ROW_PRIME_WAIT1;

            WHEN ROW_PRIME_WAIT1 =>
                state_next <= ROW_PRIME_WAIT2;

            WHEN ROW_PRIME_WAIT2 =>

                c := unsigned(clahe_in_d);

                CASE prime_cnt IS
                    WHEN 0 =>
                        w0_next <= c;
                    WHEN 1 =>
                        w1_next <= c;
                    WHEN 2 =>
                        w2_next <= c;
                    WHEN 3 =>
                        w3_next <= c;
                    WHEN 4 =>
                        w4_next <= c;
                    WHEN OTHERS =>
                        NULL;
                END CASE;

                IF prime_cnt = 4 THEN

                    state_next <= PIX_REQ;
                ELSE
                    prime_cnt_next <= prime_cnt + 1;
                    state_next     <= ROW_PRIME_REQ;
                END IF;


            WHEN PIX_REQ =>
                in_addr := (y + RADIUS) * IN_X + (x + RADIUS + 3);  -- *** KEY FIX ***

                clahe_in_en   <= '1';
                clahe_in_addr <= STD_LOGIC_VECTOR(to_unsigned(in_addr, 14));

                state_next <= PIX_WAIT1;

            WHEN PIX_WAIT1 =>
                state_next <= PIX_WAIT2;

            WHEN PIX_WAIT2 =>

                sum_next <= resize(w0, 16) +
                            resize(w1, 16) +
                            resize(w2, 16) +
                            resize(w3, 16) +
                            resize(w4, 16);

                -- 2) shift window and insert new pixel
                c := unsigned(clahe_in_d);

                w0_next <= w1;
                w1_next <= w2;
                w2_next <= w3;
                w3_next <= w4;
                w4_next <= c;

                -- go write sum
                state_next <= WRITE_OUT;

            WHEN WRITE_OUT =>
                out_addr := y * OUT_X + x;

                conv_out_en   <= '1';
                conv_out_addr <= STD_LOGIC_VECTOR(to_unsigned(out_addr, 13));
                conv_out_d    <= STD_LOGIC_VECTOR(sum_int);

                state_next    <= NEXT_PIXEL;

            WHEN NEXT_PIXEL =>
                IF x = OUT_X - 1 THEN  -- last column of this row
                    x_next <= 0;

                    IF y = OUT_Y - 1 THEN
                        state_next <= DONE;
                    ELSE

                        y_next         <= y + 1;
                        prime_cnt_next <= 0;

                        state_next <= ROW_PRIME_REQ;
                    END IF;
                ELSE
                    x_next     <= x + 1;
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
