LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;

ENTITY bram_multi_rw IS
    GENERIC (
        DATA_WIDTH : NATURAL;
        DATA_LEN : NATURAL;
        ADDR_WIDTH : NATURAL
    );
    PORT (
        -- PORT A
        clk_a : IN STD_LOGIC;
        sel_a : IN NATURAL RANGE 0 TO 3;

        we_a_0 : IN STD_LOGIC;
        re_a_0 : IN STD_LOGIC;
        addr_a_0 : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_a_0 : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        we_a_1 : IN STD_LOGIC;
        re_a_1 : IN STD_LOGIC;
        addr_a_1 : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_a_1 : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        we_a_2 : IN STD_LOGIC;
        re_a_2 : IN STD_LOGIC;
        addr_a_2 : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_a_2 : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        we_a_3 : IN STD_LOGIC;
        re_a_3 : IN STD_LOGIC;
        addr_a_3 : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_a_3 : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        dout_a : OUT STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        -- PORT B

        clk_b : IN STD_LOGIC;
        sel_b : IN NATURAL RANGE 0 TO 3;

        we_b_0 : IN STD_LOGIC;
        re_b_0 : IN STD_LOGIC;
        addr_b_0 : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_b_0 : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        we_b_1 : IN STD_LOGIC;
        re_b_1 : IN STD_LOGIC;
        addr_b_1 : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_b_1 : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        we_b_2 : IN STD_LOGIC;
        re_b_2 : IN STD_LOGIC;
        addr_b_2 : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_b_2 : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        we_b_3 : IN STD_LOGIC;
        re_b_3 : IN STD_LOGIC;
        addr_b_3 : IN STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
        din_b_3 : IN STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

        dout_b : OUT STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0)
    );
END ENTITY;

ARCHITECTURE arch OF bram_multi_rw IS
    SIGNAL we_a : STD_LOGIC;
    SIGNAL valid_addr_a_0 : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL valid_addr_a_1 : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL valid_addr_a_2 : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL valid_addr_a_3 : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL addr_a : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL din_a : STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);

    SIGNAL we_b : STD_LOGIC;
    SIGNAL valid_addr_b_0 : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL valid_addr_b_1 : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL valid_addr_b_2 : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL valid_addr_b_3 : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL addr_b : STD_LOGIC_VECTOR(ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL din_b : STD_LOGIC_VECTOR(DATA_WIDTH - 1 DOWNTO 0);
BEGIN
    we_a <= we_a_0 WHEN sel_a = 0
        ELSE
        we_a_1 WHEN sel_a = 1
        ELSE
        we_a_2 WHEN sel_a = 2
        ELSE
        we_a_3;
    valid_addr_a_0 <= addr_a_0 WHEN (we_a_0 OR re_a_0) = '1'
        ELSE
        (OTHERS => '0');
    valid_addr_a_1 <= addr_a_1 WHEN (we_a_1 OR re_a_1) = '1'
        ELSE
        (OTHERS => '0');
    valid_addr_a_2 <= addr_a_2 WHEN (we_a_2 OR re_a_2) = '1'
        ELSE
        (OTHERS => '0');
    valid_addr_a_3 <= addr_a_3 WHEN (we_a_3 OR re_a_3) = '1'
        ELSE
        (OTHERS => '0');
    addr_a <= valid_addr_a_0 WHEN sel_a = 0
        ELSE
        valid_addr_a_1 WHEN sel_a = 1
        ELSE
        valid_addr_a_2 WHEN sel_a = 2
        ELSE
        valid_addr_a_3;
    din_a <= din_a_0 WHEN sel_a = 0
        ELSE
        din_a_1 WHEN sel_a = 1
        ELSE
        din_a_2 WHEN sel_a = 2
        ELSE
        din_a_3;

    we_b <= we_b_0 WHEN sel_b = 0
        ELSE
        we_b_1 WHEN sel_b = 1
        ELSE
        we_b_2 WHEN sel_b = 2
        ELSE
        we_b_3;
    valid_addr_b_0 <= addr_b_0 WHEN (we_b_0 OR re_b_0) = '1'
        ELSE
        (OTHERS => '0');
    valid_addr_b_1 <= addr_b_1 WHEN (we_b_1 OR re_b_1) = '1'
        ELSE
        (OTHERS => '0');
    valid_addr_b_2 <= addr_b_2 WHEN (we_b_2 OR re_b_2) = '1'
        ELSE
        (OTHERS => '0');
    valid_addr_b_3 <= addr_b_3 WHEN (we_b_3 OR re_b_3) = '1'
        ELSE
        (OTHERS => '0');
    addr_b <= valid_addr_b_0 WHEN sel_b = 0
        ELSE
        valid_addr_b_1 WHEN sel_b = 1
        ELSE
        valid_addr_b_2 WHEN sel_b = 2
        ELSE
        valid_addr_b_3;
    din_b <= din_b_0 WHEN sel_b = 0
        ELSE
        din_b_1 WHEN sel_b = 1
        ELSE
        din_b_2 WHEN sel_b = 2
        ELSE
        din_b_3;

    bram_tdp_inst : ENTITY work.bram_tdp
        GENERIC MAP(
            DATA_WIDTH => DATA_WIDTH,
            DATA_LEN => DATA_LEN,
            ADDR_WIDTH => ADDR_WIDTH
        )
        PORT MAP(
            clk_a => clk_a,
            ce_a => '1',
            we_a => we_a,
            addr_a => addr_a,
            din_a => din_a,
            dout_a => dout_a,
            clk_b => clk_b,
            ce_b => '1',
            we_b => we_b,
            addr_b => addr_b,
            din_b => din_b,
            dout_b => dout_b
        );
END ARCHITECTURE;