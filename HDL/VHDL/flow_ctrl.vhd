LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY flow_ctrl IS
    PORT (
        core_clk : IN STD_LOGIC;

        cam_gen_trg : OUT STD_LOGIC;
        cam_rdy : IN STD_LOGIC;

        vga_okay_to_swap : IN STD_LOGIC;
        vga_ram_swap_trg : OUT STD_LOGIC
    );
END ENTITY;

ARCHITECTURE arch OF flow_ctrl IS
    TYPE s_type IS (WAIT_FOR_FRAME, WAIT_FOR_VGA_FREE);
    SIGNAL s : s_type := WAIT_FOR_FRAME;
    SIGNAL s_next : s_type;

    SIGNAL vga_ram_swap_trg_next : STD_LOGIC;
BEGIN
    clk_div_inst : ENTITY work.clk_div
        GENERIC MAP(
            DIV => 60000000 -- 240 MHz to 4 Hz
        )
        PORT MAP(
            clk => core_clk,
            ce => cam_gen_trg
        );

    PROCESS (s, cam_rdy, vga_okay_to_swap) BEGIN
        s_next <= s;
        vga_ram_swap_trg_next <= '0';

        CASE s IS
            WHEN WAIT_FOR_FRAME =>
                IF cam_rdy = '1' THEN
                    s_next <= WAIT_FOR_VGA_FREE;
                END IF;
            WHEN WAIT_FOR_VGA_FREE =>
                IF vga_okay_to_swap = '1' THEN
                    s_next <= WAIT_FOR_FRAME;
                    vga_ram_swap_trg_next <= '1';
                END IF;
        END CASE;
    END PROCESS;

    PROCESS (core_clk) BEGIN
        IF rising_edge(core_clk) THEN
            s <= s_next;
            vga_ram_swap_trg <= vga_ram_swap_trg_next;
        END IF;
    END PROCESS;
END ARCHITECTURE;