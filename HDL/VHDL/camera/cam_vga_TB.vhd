LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY cam_vga_TB IS
END cam_vga_TB;

ARCHITECTURE sim OF cam_vga_TB IS
    CONSTANT PCLK_PERIOD : TIME := 166.66 ns;

    SIGNAL pclk : STD_LOGIC := '0';
    SIGNAL vsync : STD_LOGIC := '1';
    SIGNAL hsync : STD_LOGIC := '0';
    SIGNAL data : STD_LOGIC_VECTOR(7 DOWNTO 0) := (OTHERS => '0');

    SIGNAL px_byte : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL px_rdy : STD_LOGIC;
    SIGNAL frame_writing : STD_LOGIC;
    SIGNAL x : NATURAL RANGE 0 TO 127;
    SIGNAL y : NATURAL RANGE 0 TO 127;
BEGIN

    c_int : ENTITY work.cam_vga
        PORT MAP(
            pclk => pclk,
            vsync => vsync,
            hsync => hsync,
            data => data,
            px_byte => px_byte,
            px_rdy => px_rdy,
            frame_writing => frame_writing,
            x => x,
            y => y
        );

    PROCESS
    BEGIN
        pclk <= '0';
        WAIT FOR PCLK_PERIOD/2;
        pclk <= '1';
        WAIT FOR PCLK_PERIOD/2;
    END PROCESS;

    PROCESS
        VARIABLE px_value : NATURAL;
    BEGIN
        FOR frame IN 0 TO 2 LOOP
            px_value := 0;
            vsync <= '1';
            hsync <= '0';
            WAIT FOR 3 * 784 * 2 * PCLK_PERIOD;

            vsync <= '0';

            WAIT FOR 96 * 784 * 2 * PCLK_PERIOD;
            FOR row IN 0 TO 127 LOOP
                hsync <= '1';
                FOR col IN 0 TO 127 LOOP
                    IF (col MOD 2 = 0) THEN
                        data <= STD_LOGIC_VECTOR(to_unsigned(px_value, 8)); -- real pixel
                        px_value := px_value + 1;
                    ELSE
                        data <= x"ff"; -- trash byte
                    END IF;

                    WAIT FOR PCLK_PERIOD;
                END LOOP;
                hsync <= '0';
                WAIT FOR (784 - 128) * 2 * PCLK_PERIOD;
            END LOOP;
            WAIT FOR (320 - 224) * 2 * 784 * PCLK_PERIOD;
        END LOOP;
        WAIT;
    END PROCESS;

END sim;