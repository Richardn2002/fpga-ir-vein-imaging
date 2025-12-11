LIBRARY IEEE;
USE ieee.std_logic_1164.ALL;

ENTITY cam_ov7670_ctrl IS
    GENERIC (
        IGNORE_I2C_ACK : BOOLEAN := FALSE
    );
    PORT (
        clk : IN STD_LOGIC; -- 12 MHz input clock

        rst : OUT STD_LOGIC;
        pwdn : OUT STD_LOGIC;

        scl_in : IN STD_LOGIC;
        scl_out : OUT STD_LOGIC;
        sda_in : IN STD_LOGIC;
        sda_out : OUT STD_LOGIC
    );
END ENTITY;

ARCHITECTURE arch OF cam_ov7670_ctrl IS
    TYPE s_type IS (START, START_DELAY, REG_RST, REG_RST_WAIT, CONFIG, CONFIG_WAIT, DONE);
    SIGNAL s : s_type := START;
    SIGNAL s_next : s_type;

    -- delay counter
    CONSTANT TICKS_1_MS : NATURAL := 12000;
    CONSTANT TICKS_2_US : NATURAL := 24;
    --- resets on 0, increments on 1
    SIGNAL delay_cnt_en : NATURAL RANGE 0 TO 1;
    SIGNAL delay_cnt : NATURAL RANGE 0 TO TICKS_1_MS - 1;

    SIGNAL subaddr_data : STD_LOGIC_VECTOR(15 DOWNTO 0);

    -- config counter
    CONSTANT NUM_CONFIG : NATURAL := 17;
    SIGNAL config_cnt : NATURAL RANGE 0 TO NUM_CONFIG - 1;
    SIGNAL config_cnt_next : NATURAL RANGE 0 TO NUM_CONFIG - 1;

    SIGNAL i2c_trg : STD_LOGIC;
    SIGNAL i2c_rdy : STD_LOGIC;
    SIGNAL i2c_err : STD_LOGIC;
BEGIN
    pwdn <= '0';
    rst <= '1';

    -- delay counter
    PROCESS (clk, delay_cnt_en)
    BEGIN
        IF delay_cnt_en = 0 THEN
            delay_cnt <= 0;
        ELSIF rising_edge(clk) THEN
            delay_cnt <= delay_cnt + 1;
        END IF;
    END PROCESS;
    -- config counter
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            config_cnt <= config_cnt_next;
        END IF;
    END PROCESS;

    -- decide config to write
    PROCESS (s, config_cnt) BEGIN
        IF s = REG_RST THEN
            subaddr_data <= x"1280";
        ELSE
            CASE config_cnt IS
                    -- WHEN 00 => subaddr_data <= x"1101"; -- divide input clock by 2
                WHEN 00 => subaddr_data <= x"1100"; -- divide input clock by 1
                WHEN 01 => subaddr_data <= x"1200";
                WHEN 02 => subaddr_data <= x"0C04";
                WHEN 03 => subaddr_data <= x"3E19";
                WHEN 04 => subaddr_data <= x"7211";
                WHEN 05 => subaddr_data <= x"73F1";
                WHEN 06 => subaddr_data <= x"A202";
                WHEN 07 => subaddr_data <= x"172D";
                WHEN 08 => subaddr_data <= x"184D";
                WHEN 09 => subaddr_data <= x"3280";
                WHEN 10 => subaddr_data <= x"191F";
                WHEN 11 => subaddr_data <= x"1A5F";
                WHEN 12 => subaddr_data <= x"0300";
                WHEN 13 => subaddr_data <= x"1308";
                WHEN 14 => subaddr_data <= x"0700";
                WHEN 15 => subaddr_data <= x"1020"; -- assumes 24 MHz clock after division @ led 1.09 A
                WHEN 16 => subaddr_data <= x"0400";
                WHEN OTHERS => subaddr_data <= x"FFFF";
            END CASE;
        END IF;
    END PROCESS;

    -- FSM
    PROCESS (s, delay_cnt, config_cnt, i2c_rdy, i2c_err) BEGIN
        s_next <= s;
        delay_cnt_en <= 0;
        config_cnt_next <= config_cnt;
        i2c_trg <= '0';

        CASE s IS
            WHEN START =>
                s_next <= START_DELAY;
            WHEN START_DELAY =>
                delay_cnt_en <= 1;
                IF delay_cnt = TICKS_1_MS - 1 THEN
                    s_next <= REG_RST;
                    i2c_trg <= '1';
                END IF;
            WHEN REG_RST =>
                IF i2c_rdy = '1' THEN
                    s_next <= REG_RST_WAIT;
                ELSIF i2c_err = '1' THEN
                    s_next <= START;
                END IF;
            WHEN REG_RST_WAIT =>
                delay_cnt_en <= 1;
                IF delay_cnt = TICKS_1_MS - 1 THEN
                    s_next <= CONFIG;
                    config_cnt_next <= 0;
                    i2c_trg <= '1';
                END IF;
            WHEN CONFIG =>
                IF i2c_rdy = '1' THEN
                    IF config_cnt = NUM_CONFIG - 1 THEN
                        s_next <= DONE;
                    ELSE
                        s_next <= CONFIG_WAIT;
                    END IF;
                ELSIF i2c_err = '1' THEN
                    s_next <= START;
                END IF;
            WHEN CONFIG_WAIT =>
                delay_cnt_en <= 1;
                IF delay_cnt = TICKS_2_US - 1 THEN
                    s_next <= CONFIG;
                    config_cnt_next <= config_cnt + 1;
                    i2c_trg <= '1';
                END IF;
            WHEN DONE => NULL;
        END CASE;
    END PROCESS;

    -- FSM, sequential
    PROCESS (clk) BEGIN
        IF rising_edge(clk) THEN
            s <= s_next;
        END IF;
    END PROCESS;

    i2c : ENTITY work.i2c_write_master
        GENERIC MAP(
            IGNORE_ACK => IGNORE_I2C_ACK
        )
        PORT MAP(
            clk => clk,
            subaddr_data => subaddr_data,
            trg => i2c_trg,
            rdy => i2c_rdy,
            err => i2c_err,
            scl_in => scl_in,
            scl_out => scl_out,
            sda_in => sda_in,
            sda_out => sda_out
        );
END ARCHITECTURE;