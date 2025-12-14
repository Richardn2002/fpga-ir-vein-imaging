LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

USE work.constants;

ENTITY cam_test_pattern IS
    -- for every trigger, write vertical stripes at odd/even positions
    GENERIC (
        STRIP_WIDTH : NATURAL := 16;
        OUTPUT_X : NATURAL := constants.INPUT_X;
        OUTPUT_Y : NATURAL := constants.INPUT_Y;
        OUTPUT_ADDR_BITS : NATURAL := constants.INPUT_ADDR_BITS
    );
    PORT (
        pclk : IN STD_LOGIC;
        trg : IN STD_LOGIC;
        rdy : OUT STD_LOGIC;

        ram_we : OUT STD_LOGIC;
        ram_addr : OUT STD_LOGIC_VECTOR(OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        ram_d : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END ENTITY;

ARCHITECTURE arch OF cam_test_pattern IS
    TYPE s_type IS (IDLE, WRITING, WRITE_LAST);
    SIGNAL s : s_type := IDLE;
    SIGNAL s_next : s_type;
    ATTRIBUTE fsm_safe_state : STRING;
    ATTRIBUTE fsm_safe_state OF s : SIGNAL IS "power_on_state";

    SIGNAL rdy_next : STD_LOGIC;

    SIGNAL pixel_idx : NATURAL RANGE 0 TO OUTPUT_X * OUTPUT_Y - 1;
    SIGNAL pixel_idx_next : NATURAL RANGE 0 TO OUTPUT_X * OUTPUT_Y - 1;
    SIGNAL strip_at_odd : BOOLEAN := TRUE;
    SIGNAL strip_at_odd_next : BOOLEAN;
BEGIN
    PROCESS (s, trg, pixel_idx, strip_at_odd) BEGIN
        rdy_next <= '0';
        ram_we <= '0';
        ram_addr <= STD_LOGIC_VECTOR(to_unsigned(pixel_idx, OUTPUT_ADDR_BITS));
        IF (((pixel_idx MOD OUTPUT_X) / STRIP_WIDTH) MOD 2 = 0) XNOR strip_at_odd THEN
            ram_d <= (OTHERS => '1');
        ELSE
            ram_d <= (OTHERS => '0');
        END IF;
        strip_at_odd_next <= strip_at_odd;
        s_next <= s;
        pixel_idx_next <= 0;

        CASE s IS
            WHEN IDLE =>
                IF trg = '1' THEN
                    s_next <= WRITING;
                    strip_at_odd_next <= NOT strip_at_odd;
                END IF;
            WHEN WRITING =>
                ram_we <= '1';
                pixel_idx_next <= pixel_idx + 1;
                IF pixel_idx = OUTPUT_X * OUTPUT_Y - 2 THEN
                    s_next <= WRITE_LAST;
                END IF;
            WHEN WRITE_LAST =>
                rdy_next <= '1';
                ram_we <= '1';
                s_next <= IDLE;
        END CASE;
    END PROCESS;

    PROCESS (pclk, s_next, pixel_idx_next, strip_at_odd_next) BEGIN
        IF rising_edge(pclk) THEN
            s <= s_next;
            rdy <= rdy_next;
            pixel_idx <= pixel_idx_next;
            strip_at_odd <= strip_at_odd_next;
        END IF;
    END PROCESS;
END ARCHITECTURE;