LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

USE work.constants;

ENTITY CLAHE_mappings IS
    PORT (
        clk : IN STD_LOGIC; -- clock input
        trg : IN STD_LOGIC; -- high for one period to start
        rdy : OUT STD_LOGIC; -- high for one period on completion

        img_in_en : OUT STD_LOGIC;
        img_in_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_PATCH_ADDR_BITS - 1 DOWNTO 0);
        img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        hist_mapping_inout_ren : OUT STD_LOGIC;
        hist_mapping_inout_wen : OUT STD_LOGIC;
        hist_mapping_inout_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS - 1 DOWNTO 0);
        hist_mapping_inout_din : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        hist_mapping_inout_dout : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END CLAHE_mappings;