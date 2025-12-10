LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

USE work.constants;

ENTITY cam_n_bram IS
    GENERIC (
        INIT_USE_0 : BOOLEAN;
        OUTPUT_X : NATURAL;
        OUTPUT_Y : NATURAL;
        ADDR_BITS : NATURAL := constants.INPUT_ADDR_BITS
    );
    PORT (
        cam_clk : IN STD_LOGIC;
        trg : IN STD_LOGIC;

        cam_we : IN STD_LOGIC;
        cam_x : IN NATURAL RANGE 0 TO constants.INPUT_X - 1;
        cam_y : IN NATURAL RANGE 0 TO constants.INPUT_Y - 1;
        cam_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        bram_we_0 : OUT STD_LOGIC;
        bram_addr_0 : OUT STD_LOGIC_VECTOR(ADDR_BITS - 1 DOWNTO 0);
        bram_d_0 : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);

        bram_we_1 : OUT STD_LOGIC;
        bram_addr_1 : OUT STD_LOGIC_VECTOR(ADDR_BITS - 1 DOWNTO 0);
        bram_d_1 : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END ENTITY;

ARCHITECTURE arch OF cam_n_bram IS
    SIGNAL use_0 : BOOLEAN := INIT_USE_0;
    SIGNAL use_0_next : BOOLEAN;
    SIGNAL valid_addr : STD_LOGIC_VECTOR(ADDR_BITS - 1 DOWNTO 0);
BEGIN
    -- combinatorial
    --- broadcast connections
    bram_d_0 <= cam_d;
    bram_d_1 <= cam_d;
    bram_addr_0 <= valid_addr;
    bram_addr_1 <= valid_addr;
    --- only use addr value when module drives write enable
    PROCESS (cam_we, cam_x, cam_y) BEGIN
        IF cam_we = '1' THEN
            IF cam_x >= OUTPUT_X OR cam_y >= OUTPUT_Y THEN
                -- out of selected output range, route to last entry of ram
                valid_addr <= (OTHERS => '1');
            ELSE
                valid_addr <= STD_LOGIC_VECTOR(to_unsigned(cam_x, ADDR_BITS) + resize(to_unsigned(cam_y, ADDR_BITS) * OUTPUT_Y, ADDR_BITS));
            END IF;
        ELSE
            valid_addr <= (OTHERS => '0');
        END IF;
    END PROCESS;
    --- MUX to select BRAM to write to
    PROCESS (use_0, cam_we) BEGIN
        IF use_0 THEN
            bram_we_0 <= cam_we;
            bram_we_1 <= '0';
        ELSE
            bram_we_0 <= '0';
            bram_we_1 <= cam_we;
        END IF;
    END PROCESS;
    --- swaps on trigger
    PROCESS (trg, use_0) BEGIN
        IF trg = '1' THEN
            use_0_next <= NOT use_0;
        ELSE
            use_0_next <= use_0;
        END IF;
    END PROCESS;

    -- sequential
    PROCESS (cam_clk) BEGIN
        IF rising_edge(cam_clk) THEN
            use_0 <= use_0_next;
        END IF;
    END PROCESS;
END ARCHITECTURE;