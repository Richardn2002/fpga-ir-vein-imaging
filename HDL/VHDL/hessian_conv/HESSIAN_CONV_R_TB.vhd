LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY hessian_conv_r_tb IS
END hessian_conv_r_tb;

ARCHITECTURE sim OF hessian_conv_r_tb IS

    SIGNAL clk : STD_LOGIC := '0';
    SIGNAL trg : STD_LOGIC := '0';
    SIGNAL rdy : STD_LOGIC;

    SIGNAL clahe_in_en : STD_LOGIC;
    SIGNAL clahe_in_addr : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL clahe_in_d : STD_LOGIC_VECTOR(7 DOWNTO 0);

    SIGNAL conv_out_en : STD_LOGIC;
    SIGNAL conv_out_addr : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL conv_out_d : STD_LOGIC_VECTOR(15 DOWNTO 0);
    TYPE img_t IS ARRAY (0 TO 96 * 96 - 1) OF unsigned(7 DOWNTO 0);
    SIGNAL img_ram : img_t := (OTHERS => (OTHERS => '0'));

    TYPE out_t IS ARRAY (0 TO 96 * 96 - 1) OF signed(15 DOWNTO 0);
    SIGNAL out_ram : out_t := (OTHERS => (OTHERS => '0'));

BEGIN

    clk <= NOT clk AFTER 5 ns;
    dut : ENTITY work.HESSIAN_conv_r
        PORT MAP(
            clk => clk,
            trg => trg,
            rdy => rdy,
            clahe_in_en => clahe_in_en,
            clahe_in_addr => clahe_in_addr,
            clahe_in_d => clahe_in_d,
            conv_out_en => conv_out_en,
            conv_out_addr => conv_out_addr,
            conv_out_d => conv_out_d
        );
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF clahe_in_en = '1' THEN
                clahe_in_d <= STD_LOGIC_VECTOR(img_ram(to_integer(unsigned(clahe_in_addr))));
            END IF;
        END IF;
    END PROCESS;
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF conv_out_en = '1' THEN
                out_ram(to_integer(unsigned(conv_out_addr)))
                <= signed(conv_out_d);
            END IF;
        END IF;
    END PROCESS;
    stim : PROCESS
    BEGIN

        FOR y IN 0 TO 95 LOOP
            FOR x IN 0 TO 95 LOOP
                img_ram(y * 96 + x) <= to_unsigned(x, 8);
            END LOOP;
        END LOOP;

        WAIT FOR 20 ns;

        -- Start DUT
        trg <= '1';
        WAIT FOR 20 ns;
        trg <= '0';
        WAIT UNTIL rdy = '1';

        -- Stop sim
        WAIT FOR 50 ns;
        ASSERT false REPORT "Simulation Finished" SEVERITY failure;
    END PROCESS;

END sim;