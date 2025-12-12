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
BEGIN
    bram_swapper_r_inst : ENTITY work.bram_swapper_r
        GENERIC MAP(
            INIT_USE_0 => INIT_USE_0,
            ADDR_BITS => constants.HESSIAN_OUTPUT_ADDR_BITS,
            DATA_BITS => 8
        )
        PORT MAP(
            clk => vga_clk,
            trg => trg,
            re => vga_re,
            addr => vga_addr,
            d => vga_d,
            bram_we_0 => bram_we_0,
            bram_addr_0 => bram_addr_0,
            bram_d_0 => bram_d_0,
            bram_we_1 => bram_we_1,
            bram_addr_1 => bram_addr_1,
            bram_d_1 => bram_d_1
        );
END ARCHITECTURE;