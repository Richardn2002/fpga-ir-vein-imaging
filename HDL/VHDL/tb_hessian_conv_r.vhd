library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity tb_hessian_conv_r is
end tb_hessian_conv_r;

architecture sim of tb_hessian_conv_r is

    signal clk  : std_logic := '0';
    signal trg  : std_logic := '0';
    signal rdy  : std_logic;

    signal clahe_in_en   : std_logic;
    signal clahe_in_addr : std_logic_vector(13 downto 0);
    signal clahe_in_d    : std_logic_vector(7 downto 0);

    signal conv_out_en   : std_logic;
    signal conv_out_addr : std_logic_vector(13 downto 0);
    signal conv_out_d    : std_logic_vector(15 downto 0);


    type img_t is array (0 to 96*96-1) of unsigned(7 downto 0);
    signal img_ram : img_t := (others => (others => '0'));

    type out_t is array (0 to 96*96-1) of signed(15 downto 0);
    signal out_ram : out_t := (others => (others => '0'));

begin

    clk <= not clk after 5 ns;


    dut: entity work.HESSIAN_conv_r
        port map (
            clk => clk,
            trg => trg,
            rdy => rdy,
            clahe_in_en   => clahe_in_en,
            clahe_in_addr => clahe_in_addr,
            clahe_in_d    => clahe_in_d,
            conv_out_en   => conv_out_en,
            conv_out_addr => conv_out_addr,
            conv_out_d    => conv_out_d
        );


    process(clk)
    begin
        if rising_edge(clk) then
            if clahe_in_en = '1' then
                clahe_in_d <= std_logic_vector(img_ram(to_integer(unsigned(clahe_in_addr))));
            end if;
        end if;
    end process;


    process(clk)
    begin
        if rising_edge(clk) then
            if conv_out_en = '1' then
                out_ram(to_integer(unsigned(conv_out_addr)))
                    <= signed(conv_out_d);
            end if;
        end if;
    end process;


    stim: process
    begin

        for y in 0 to 95 loop
            for x in 0 to 95 loop
                img_ram(y*96 + x) <= to_unsigned(x,8);
            end loop;
        end loop;

        wait for 20 ns;

        -- Start DUT
        trg <= '1';
        wait for 20 ns;
        trg <= '0';


        wait until rdy = '1';

        -- Stop sim
        wait for 50 ns;
        assert false report "Simulation Finished" severity failure;
    end process;

end sim;
