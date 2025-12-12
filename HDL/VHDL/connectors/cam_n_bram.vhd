LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

USE work.constants;

ENTITY cam_n_bram IS
    GENERIC (
        INIT_USE_0 : BOOLEAN;
        OUTPUT_X : NATURAL := constants.INPUT_X;
        OUTPUT_Y : NATURAL := constants.INPUT_Y;
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
    SIGNAL valid_we : STD_LOGIC;
    SIGNAL valid_addr : STD_LOGIC_VECTOR(ADDR_BITS - 1 DOWNTO 0);
BEGIN
    --- only use addr value when module drives write enable
    PROCESS (cam_we, cam_x, cam_y) BEGIN
        IF cam_we = '1' AND cam_x < OUTPUT_X AND cam_y < OUTPUT_Y THEN
            valid_we <= '1';
            valid_addr <= STD_LOGIC_VECTOR(to_unsigned(cam_x, ADDR_BITS) + resize(to_unsigned(cam_y, ADDR_BITS) * OUTPUT_X, ADDR_BITS));
        ELSE
            valid_we <= '0';
            valid_addr <= (OTHERS => '0');
        END IF;
    END PROCESS;

    bram_swapper_w_inst : ENTITY work.bram_swapper_w
        GENERIC MAP(
            INIT_USE_0 => INIT_USE_0,
            ADDR_BITS => ADDR_BITS,
            DATA_BITS => 8
        )
        PORT MAP(
            clk => cam_clk,
            trg => trg,
            we => valid_we,
            addr => valid_addr,
            d => cam_d,
            bram_we_0 => bram_we_0,
            bram_addr_0 => bram_addr_0,
            bram_d_0 => bram_d_0,
            bram_we_1 => bram_we_1,
            bram_addr_1 => bram_addr_1,
            bram_d_1 => bram_d_1
        );
END ARCHITECTURE;