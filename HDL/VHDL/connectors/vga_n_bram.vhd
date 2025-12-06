LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

USE work.constants;

ENTITY vga_n_bram IS
    GENERIC (
        CONSTANT INIT_USE_0 : BOOLEAN
    );
    PORT (
        vga_clk : IN STD_LOGIC;
        trg : IN STD_LOGIC;

        vga_re : IN STD_LOGIC;
        vga_addr : IN STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        vga_d : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);

        bram_we_0 : OUT STD_LOGIC;
        bram_addr_0 : OUT STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        bram_d_0 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        bram_we_1 : OUT STD_LOGIC;
        bram_addr_1 : OUT STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        bram_d_1 : IN STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END ENTITY;

ARCHITECTURE arch OF vga_n_bram IS
    SIGNAL use_0 : BOOLEAN := INIT_USE_0;
    SIGNAL use_0_next : BOOLEAN;
    SIGNAL valid_addr : STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
BEGIN
    -- combinatorial
    --- broadcast connections
    bram_we_0 <= '0';
    bram_we_1 <= '0';
    bram_addr_0 <= valid_addr;
    bram_addr_1 <= valid_addr;
    --- only use addr value when module drives read enable
    PROCESS (vga_re, vga_addr) BEGIN
        IF vga_re = '1' THEN
            valid_addr <= vga_addr;
        ELSE
            valid_addr <= (OTHERS => '0');
        END IF;
    END PROCESS;
    --- MUX to select BRAM to read from
    PROCESS (use_0, bram_d_0, bram_d_1) BEGIN
        IF use_0 THEN
            vga_d <= bram_d_0;
        ELSE
            vga_d <= bram_d_1;
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
    PROCESS (vga_clk) BEGIN
        IF rising_edge(vga_clk) THEN
            use_0 <= use_0_next;
        END IF;
    END PROCESS;
END ARCHITECTURE;