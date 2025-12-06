LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;

ENTITY bram_tdp IS
    GENERIC (
        DATA_WIDTH : NATURAL;
        DATA_LEN : NATURAL;
        ADDR_WIDTH : NATURAL
    );
    PORT (
        clk_a : IN STD_LOGIC;
        ce_a : IN STD_LOGIC;
        we_a : IN STD_LOGIC;
        addr_a : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_a : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);
        dout_a : OUT STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        clk_b : IN STD_LOGIC;
        ce_b : IN STD_LOGIC;
        we_b : IN STD_LOGIC;
        addr_b : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_b : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);
        dout_b : OUT STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0)
    );
END ENTITY bram_tdp;

ARCHITECTURE arch OF bram_tdp IS
    TYPE ram_type IS ARRAY (0 TO DATA_LEN - 1) OF STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);
    -- SHARED VARIABLE according to https://docs.amd.com/r/en-US/ug901-vivado-synthesis/Dual-Port-Block-RAM-with-Two-Write-Ports-in-Read-First-Mode-VHDL
    SHARED VARIABLE ram_array : ram_type;
BEGIN
    PROCESS (clk_a)
    BEGIN
        IF rising_edge(clk_a) AND ce_a = '1' THEN
            -- read out old value first, otherwise the BRAM internal output registers can not be inferred
            dout_a <= ram_array(to_integer(unsigned(addr_a)));
            IF we_a = '1' THEN
                ram_array(to_integer(unsigned(addr_a))) := din_a;
            END IF;
        END IF;
    END PROCESS;

    PROCESS (clk_b)
    BEGIN
        IF rising_edge(clk_b) AND ce_b = '1' THEN
            dout_b <= ram_array(to_integer(unsigned(addr_b)));
            IF we_b = '1' THEN
                ram_array(to_integer(unsigned(addr_b))) := din_b;
            END IF;
        END IF;
    END PROCESS;
END ARCHITECTURE arch;