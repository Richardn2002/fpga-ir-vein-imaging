LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

USE work.constants;

ENTITY CLAHE_output IS
    PORT (
        clk : IN STD_LOGIC; -- clock input
        trg : IN STD_LOGIC; -- high for one period to start
        rdy : OUT STD_LOGIC; -- high for one period on completion

        img_in_en : OUT STD_LOGIC;
        img_in_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_IMG_ADDR_BITS - 1 DOWNTO 0);
        img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        mapping_in_en : OUT STD_LOGIC;
        mapping_in_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS - 1 DOWNTO 0);
        mapping_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        clahe_out_en : OUT STD_LOGIC;
        clahe_out_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        clahe_out_d : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END CLAHE_output;