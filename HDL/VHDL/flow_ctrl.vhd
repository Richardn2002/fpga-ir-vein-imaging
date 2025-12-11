LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY flow_ctrl IS
    PORT (
        core_clk : IN STD_LOGIC;

        cam_frame_writing : IN STD_LOGIC;
        cam_ram_swap_trg : OUT STD_LOGIC;

        vga_okay_to_swap : IN STD_LOGIC;
        vga_ram_swap_trg : OUT STD_LOGIC
    );
END ENTITY;

ARCHITECTURE arch OF flow_ctrl IS
    SIGNAL new_frame_incoming : BOOLEAN := FALSE;
    SIGNAL cam_writing_frame_last : STD_LOGIC := '0';

    TYPE s_type IS (WAIT_FOR_FRAME, COMPUTING, WAIT_FOR_VGA_FREE);
    SIGNAL s : s_type := WAIT_FOR_FRAME;
    SIGNAL s_next : s_type;

    SIGNAL cam_ram_swap_trg_next : STD_LOGIC;
    SIGNAL vga_ram_swap_trg_next : STD_LOGIC;
BEGIN
    PROCESS (core_clk) BEGIN
        IF rising_edge(core_clk) THEN
            cam_writing_frame_last <= cam_frame_writing;
            IF cam_writing_frame_last = '0' AND cam_frame_writing = '1' THEN
                new_frame_incoming <= TRUE;
            END IF;
            IF s = COMPUTING THEN
                new_frame_incoming <= FALSE;
            END IF;
        END IF;
    END PROCESS;

    PROCESS (s, new_frame_incoming, cam_frame_writing, vga_okay_to_swap) BEGIN
        s_next <= s;
        cam_ram_swap_trg_next <= '0';
        vga_ram_swap_trg_next <= '0';

        CASE s IS
            WHEN WAIT_FOR_FRAME =>
                IF new_frame_incoming AND cam_frame_writing = '0' THEN
                    s_next <= COMPUTING;
                    cam_ram_swap_trg_next <= '1';
                END IF;
            WHEN COMPUTING =>
                -- TODO
                s_next <= WAIT_FOR_VGA_FREE;
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
            cam_ram_swap_trg <= cam_ram_swap_trg_next;
            vga_ram_swap_trg <= vga_ram_swap_trg_next;
        END IF;
    END PROCESS;
END ARCHITECTURE;