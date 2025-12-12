LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

USE work.constants;

ENTITY clahe_n_bram IS
    GENERIC (
        INIT_READER_MAPPING : BOOLEAN
    );
    PORT (
        core_clk : IN STD_LOGIC;
        clahe_reader_swap_trg : IN STD_LOGIC;

        bram_re : OUT STD_LOGIC;
        bram_addr : OUT STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
        bram_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        clahe_mapping_re : IN STD_LOGIC;
        clahe_mapping_addr : IN STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
        clahe_mapping_d : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);

        clahe_output_re : IN STD_LOGIC;
        clahe_output_addr : IN STD_LOGIC_VECTOR(constants.INPUT_ADDR_BITS - 1 DOWNTO 0);
        clahe_output_d : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END ENTITY;

ARCHITECTURE arch OF clahe_n_bram IS
    SIGNAL reader_mapping : BOOLEAN := INIT_READER_MAPPING;
    SIGNAL reader_mapping_next : BOOLEAN;
BEGIN
    -- combinatorial
    --- broadcast connections
    clahe_mapping_d <= bram_d;
    clahe_output_d <= bram_d;
    --- MUX to select who to read the BRAM
    PROCESS (reader_mapping, clahe_mapping_re, clahe_mapping_addr, clahe_output_re, clahe_output_addr) BEGIN
        IF reader_mapping THEN
            bram_re <= clahe_mapping_re;
            bram_addr <= clahe_mapping_addr;
        ELSE
            bram_re <= clahe_output_re;
            bram_addr <= clahe_output_addr;
        END IF;
    END PROCESS;
    --- swaps on trigger
    PROCESS (clahe_reader_swap_trg, reader_mapping) BEGIN
        IF clahe_reader_swap_trg = '1' THEN
            reader_mapping_next <= NOT reader_mapping;
        ELSE
            reader_mapping_next <= reader_mapping;
        END IF;
    END PROCESS;

    -- sequential
    PROCESS (core_clk) BEGIN
        IF rising_edge(core_clk) THEN
            reader_mapping <= reader_mapping_next;
        END IF;
    END PROCESS;
END ARCHITECTURE;