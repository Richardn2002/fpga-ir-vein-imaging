LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY bram_swapper_r IS
    GENERIC (
        CONSTANT INIT_USE_0 : BOOLEAN;
        ADDR_BITS : NATURAL;
        DATA_BITS : NATURAL
    );
    PORT (
        clk : IN STD_LOGIC;
        trg : IN STD_LOGIC;

        re : IN STD_LOGIC;
        addr : IN STD_LOGIC_VECTOR(ADDR_BITS - 1 DOWNTO 0);
        d : OUT STD_LOGIC_VECTOR(DATA_BITS - 1 DOWNTO 0);

        bram_we_0 : OUT STD_LOGIC;
        bram_addr_0 : OUT STD_LOGIC_VECTOR(ADDR_BITS - 1 DOWNTO 0);
        bram_d_0 : IN STD_LOGIC_VECTOR(DATA_BITS - 1 DOWNTO 0);

        bram_we_1 : OUT STD_LOGIC;
        bram_addr_1 : OUT STD_LOGIC_VECTOR(ADDR_BITS - 1 DOWNTO 0);
        bram_d_1 : IN STD_LOGIC_VECTOR(DATA_BITS - 1 DOWNTO 0)
    );
END ENTITY;

ARCHITECTURE arch OF bram_swapper_r IS
    SIGNAL use_0 : BOOLEAN := INIT_USE_0;
    SIGNAL use_0_next : BOOLEAN;
    SIGNAL valid_addr : STD_LOGIC_VECTOR(ADDR_BITS - 1 DOWNTO 0);
BEGIN
    -- combinatorial
    --- broadcast connections
    bram_we_0 <= '0';
    bram_we_1 <= '0';
    bram_addr_0 <= valid_addr;
    bram_addr_1 <= valid_addr;
    --- only use addr value when module drives read enable
    PROCESS (re, addr) BEGIN
        IF re = '1' THEN
            valid_addr <= addr;
        ELSE
            valid_addr <= (OTHERS => '0');
        END IF;
    END PROCESS;
    --- MUX to select BRAM to read from
    PROCESS (use_0, bram_d_0, bram_d_1) BEGIN
        IF use_0 THEN
            d <= bram_d_0;
        ELSE
            d <= bram_d_1;
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
    PROCESS (clk) BEGIN
        IF rising_edge(clk) THEN
            use_0 <= use_0_next;
        END IF;
    END PROCESS;
END ARCHITECTURE;