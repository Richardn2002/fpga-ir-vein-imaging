LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY i2c_write_master IS
    GENERIC (
        ADDR : STD_LOGIC_VECTOR(6 DOWNTO 0) := b"0100001";
        IGNORE_ACK : BOOLEAN := FALSE
    );
    PORT (
        clk : IN STD_LOGIC; -- 12 MHz clock input

        subaddr_data : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
        trg : IN STD_LOGIC;
        rdy : OUT STD_LOGIC;
        err : OUT STD_LOGIC;

        scl_in : IN STD_LOGIC;
        scl_out : OUT STD_LOGIC := '1';
        sda_in : IN STD_LOGIC;
        sda_out : OUT STD_LOGIC := '1'
    );
END i2c_write_master;

ARCHITECTURE arch OF i2c_write_master IS
    -- delay counter
    CONSTANT TICKS_FIVE_US : NATURAL := 60;
    --- resets on 0, increments on 1
    SIGNAL cnt_en : NATURAL RANGE 0 TO 1;
    SIGNAL cnt : NATURAL RANGE 0 TO 255;

    TYPE prot_s_type IS (
        IDLE, START, P,
        ADDR_W, WRITE_SUB_ADDR, WRITE_DATA
    );
    SIGNAL prot_s : prot_s_type := IDLE;
    SIGNAL prot_s_next : prot_s_type;

    -- bit counter
    SIGNAL bit_cnt : NATURAL RANGE 0 TO 8;
    SIGNAL bit_cnt_next : NATURAL RANGE 0 TO 9;
    TYPE s_type IS (
        IDLE,
        S_SDA_DOWN, S_SDA_HOLD_DOWN, S_SCL_DOWN,
        W_DATA_UPDATE, W_SCL_UP, W_SCL_HOLD_UP, W_SCL_DOWN,
        R_SCL_HOLD_DOWN, R_SCL_UP, R_DATA_SAMPLE, R_SCL_DOWN,
        P_SCL_DOWN, P_SCL_HOLD_DOWN, P_SDA_DOWN, P_SCL_UP, P_SCL_HOLD_UP, P_SDA_UP
    );
    SIGNAL s : s_type := IDLE;
    SIGNAL s_next : s_type;

    SIGNAL rdy_state : BOOLEAN;
    SIGNAL rdy_state_next : BOOLEAN;
    SIGNAL rdy_next : STD_LOGIC;
    SIGNAL err_next : STD_LOGIC;

    -- bit to output during ADDR_W/WRITE
    SIGNAL out_bit : STD_LOGIC;
BEGIN
    -- delay counter
    PROCESS (clk, cnt_en)
    BEGIN
        IF cnt_en = 0 THEN
            cnt <= 0;
        ELSIF rising_edge(clk) THEN
            cnt <= cnt + 1;
        END IF;
    END PROCESS;

    -- sequential
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            prot_s <= prot_s_next;
            bit_cnt <= bit_cnt_next;
            s <= s_next;

            rdy_state <= rdy_state_next;
            rdy <= rdy_next;
            err <= err_next;
        END IF;
    END PROCESS;

    -- combinatorial
    --- what value to drive SDA with
    PROCESS (prot_s, bit_cnt, subaddr_data)
    BEGIN
        CASE prot_s IS
            WHEN ADDR_W =>
                IF bit_cnt < 7 THEN
                    out_bit <= ADDR(6 - bit_cnt);
                ELSE
                    out_bit <= '0';
                END IF;
            WHEN WRITE_SUB_ADDR =>
                out_bit <= subaddr_data(15 - bit_cnt);
            WHEN WRITE_DATA =>
                out_bit <= subaddr_data(7 - bit_cnt);
            WHEN OTHERS =>
                -- don't care
                out_bit <= '0';
        END CASE;
    END PROCESS;

    --- FSM
    PROCESS (bit_cnt, rdy_state, prot_s, s, trg, sda_in, cnt, scl_in, out_bit)
    BEGIN
        rdy_state_next <= rdy_state;
        rdy_next <= '0';
        err_next <= '0';
        cnt_en <= 0;
        bit_cnt_next <= bit_cnt;
        prot_s_next <= prot_s;
        s_next <= s;

        -- need to set: scl_out, sda_out
        -- states: prot_s_next, s_next
        -- has default value: cnt_en, bit_cnt_next, rdy_state_next, rdy_next, err_next
        CASE s IS
            WHEN IDLE =>
                scl_out <= '1';
                sda_out <= '1';
                rdy_state_next <= FALSE;

                IF trg = '1' THEN
                    prot_s_next <= START;
                    s_next <= S_SDA_DOWN;
                END IF;
            WHEN S_SDA_DOWN =>
                scl_out <= '1';
                sda_out <= '0';

                IF sda_in = '0' THEN
                    s_next <= S_SDA_HOLD_DOWN;
                END IF;
            WHEN S_SDA_HOLD_DOWN =>
                scl_out <= '1';
                sda_out <= '0';
                cnt_en <= 1;

                IF cnt = TICKS_FIVE_US / 2 - 1 THEN
                    s_next <= S_SCL_DOWN;
                END IF;
            WHEN S_SCL_DOWN =>
                scl_out <= '0';
                sda_out <= '0';

                IF scl_in = '0' THEN
                    CASE prot_s IS
                        WHEN START =>
                            -- start complete
                            prot_s_next <= ADDR_W;
                            s_next <= W_DATA_UPDATE;
                            bit_cnt_next <= 0;
                        WHEN OTHERS =>
                            -- impossible
                            NULL;
                    END CASE;
                END IF;
            WHEN W_DATA_UPDATE =>
                scl_out <= '0';
                sda_out <= out_bit;
                cnt_en <= 1;

                IF cnt = TICKS_FIVE_US - 1 THEN
                    s_next <= W_SCL_UP;
                END IF;
            WHEN W_SCL_UP =>
                scl_out <= '1';
                sda_out <= out_bit;

                IF scl_in = '1' THEN
                    s_next <= W_SCL_HOLD_UP;
                END IF;
            WHEN W_SCL_HOLD_UP =>
                scl_out <= '1';
                sda_out <= out_bit;
                cnt_en <= 1;

                IF cnt = TICKS_FIVE_US - 1 THEN
                    s_next <= W_SCL_DOWN;
                END IF;
            WHEN W_SCL_DOWN =>
                scl_out <= '0';
                sda_out <= out_bit;

                IF scl_in = '0' THEN
                    IF bit_cnt = 8 THEN
                        -- master ACK just sent
                        prot_s_next <= P;
                        s_next <= P_SDA_DOWN;
                    ELSIF bit_cnt = 7 THEN
                        -- prepare to receive ACK
                        s_next <= R_SCL_HOLD_DOWN;
                    ELSE
                        -- repeat to send next bit (R/W included)
                        s_next <= W_DATA_UPDATE;
                        bit_cnt_next <= bit_cnt + 1;
                    END IF;
                END IF;
            WHEN R_SCL_HOLD_DOWN =>
                scl_out <= '0';
                sda_out <= '1'; -- release SDA
                cnt_en <= 1;

                IF cnt = TICKS_FIVE_US - 1 THEN
                    s_next <= R_SCL_UP;
                END IF;
            WHEN R_SCL_UP =>
                scl_out <= '1';
                sda_out <= '1'; -- release SDA

                IF scl_in = '1' THEN
                    s_next <= R_DATA_SAMPLE;
                END IF;
            WHEN R_DATA_SAMPLE =>
                scl_out <= '1';
                sda_out <= '1'; -- release SDA
                cnt_en <= 1;

                IF cnt = TICKS_FIVE_US - 1 THEN
                    -- sample SDA
                    IF sda_in = '1' AND NOT IGNORE_ACK THEN
                        -- ACK not received
                        rdy_state_next <= FALSE;
                        prot_s_next <= P;
                        s_next <= P_SCL_DOWN;
                    ELSE
                        -- ACK received
                        s_next <= R_SCL_DOWN;
                    END IF;
                END IF;
            WHEN R_SCL_DOWN =>
                scl_out <= '0';
                sda_out <= '1'; -- release SDA

                IF scl_in = '0' THEN
                    CASE prot_s IS
                        WHEN ADDR_W =>
                            -- ACK from slave read
                            prot_s_next <= WRITE_SUB_ADDR;
                            s_next <= W_DATA_UPDATE;
                            bit_cnt_next <= 0;
                        WHEN WRITE_SUB_ADDR =>
                            -- ACK from slave read
                            prot_s_next <= WRITE_DATA;
                            s_next <= W_DATA_UPDATE;
                            bit_cnt_next <= 0;
                        WHEN WRITE_DATA =>
                            -- ACK from slave read
                            rdy_state_next <= TRUE;
                            prot_s_next <= P;
                            s_next <= P_SCL_DOWN;
                        WHEN OTHERS =>
                            -- impossible
                            NULL;
                    END CASE;
                END IF;
            WHEN P_SCL_DOWN =>
                scl_out <= '0';
                sda_out <= '1'; -- don't care SDA

                IF scl_in = '0' THEN
                    s_next <= P_SCL_HOLD_DOWN;
                END IF;
            WHEN P_SCL_HOLD_DOWN =>
                scl_out <= '0';
                sda_out <= '1'; -- don't care SDA
                cnt_en <= 1;

                IF cnt = TICKS_FIVE_US / 2 - 1 THEN
                    s_next <= P_SDA_DOWN;
                END IF;
            WHEN P_SDA_DOWN =>
                scl_out <= '0';
                sda_out <= '0';

                IF sda_in = '0' THEN
                    s_next <= P_SCL_UP;
                END IF;
            WHEN P_SCL_UP =>
                scl_out <= '1';
                sda_out <= '0';

                IF scl_in = '1' THEN
                    s_next <= P_SCL_HOLD_UP;
                END IF;
            WHEN P_SCL_HOLD_UP =>
                scl_out <= '1';
                sda_out <= '0';
                cnt_en <= 1;

                IF cnt = TICKS_FIVE_US / 2 - 1 THEN
                    s_next <= P_SDA_UP;
                END IF;
            WHEN P_SDA_UP =>
                scl_out <= '1';
                sda_out <= '1';

                IF sda_in = '1' THEN
                    prot_s_next <= IDLE;
                    s_next <= IDLE;

                    IF rdy_state THEN
                        rdy_next <= '1';
                    ELSE
                        err_next <= '1';
                    END IF;
                END IF;
            WHEN OTHERS =>
                scl_out <= '1';
                sda_out <= '1';
        END CASE;
    END PROCESS;
END ARCHITECTURE;