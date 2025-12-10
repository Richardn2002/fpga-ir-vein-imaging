LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY cam_vga IS
    PORT (
        pclk : IN STD_LOGIC; -- Pixel clock from camera
        vsync : IN STD_LOGIC; -- Frame indicator
        hsync : IN STD_LOGIC; -- Row on indicator 
        data : IN STD_LOGIC_VECTOR(7 DOWNTO 0); -- Pixel byte from camera

        px_byte : OUT STD_LOGIC_VECTOR(7 DOWNTO 0) := (OTHERS => '0');-- Output pixel byte
        px_rdy : OUT STD_LOGIC := '0'; -- High for 1 cycle when new pixel available
        frame_writing : OUT STD_LOGIC := '0'; -- high during a frame

        x : OUT NATURAL RANGE 0 TO 127; -- X
        y : OUT NATURAL RANGE 0 TO 127 -- Y
    );
END cam_vga;

ARCHITECTURE arch OF cam_vga IS
    SIGNAL x_cnt : NATURAL RANGE 0 TO 127 := 0;
    SIGNAL y_cnt : NATURAL RANGE 0 TO 127 := 0;

    SIGNAL vsync_prev : STD_LOGIC := '1';
    SIGNAL hsync_prev : STD_LOGIC := '0';

    SIGNAL px_rdy_toggle : STD_LOGIC := '0';
BEGIN
    PROCESS (pclk)
    BEGIN
        IF rising_edge(pclk) THEN
            px_rdy <= '0';

            IF (vsync = '0') THEN -- if active frame
                IF (hsync = '1') THEN -- if active row
                    px_byte <= data; -- sample data

                    IF (px_rdy_toggle = '0') THEN
                        px_rdy <= '1'; -- mark ready 
                        x_cnt <= x_cnt + 1; -- next column
                    ELSE
                        px_rdy <= '0';
                    END IF;

                    px_rdy_toggle <= NOT px_rdy_toggle;
                END IF;
            END IF;

            IF (vsync_prev = '1' AND vsync = '0') THEN --vsync falling edge > frame started                
                y_cnt <= 0;
                x_cnt <= 0;
            ELSIF (vsync_prev = '0' AND vsync = '1') THEN --vsync rising edge > frame ended           
                y_cnt <= 0;
                x_cnt <= 0;
                frame_writing <= '0';
            END IF;

            IF (hsync_prev = '0' AND hsync = '1') THEN --hsync rising edge > row started                
                x_cnt <= 0; --reset column
                IF y_cnt = 0 THEN
                    frame_writing <= '1';
                END IF;
            ELSIF (hsync_prev = '1' AND hsync = '0') THEN --hsync falling edge > row ended              
                y_cnt <= y_cnt + 1; --next row
            END IF;

            vsync_prev <= vsync;
            hsync_prev <= hsync;
        END IF;
    END PROCESS;

    x <= x_cnt;
    y <= y_cnt;
END arch;