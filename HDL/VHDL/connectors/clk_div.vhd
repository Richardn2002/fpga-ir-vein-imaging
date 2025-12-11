LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY clk_div IS
    GENERIC (
        DIV : NATURAL
    );
    PORT (
        clk : IN STD_LOGIC;
        ce : OUT STD_LOGIC
    );
END ENTITY;

ARCHITECTURE arch OF clk_div IS
    SIGNAL cnt : NATURAL RANGE 0 TO DIV - 1 := DIV - 1;
BEGIN
    PROCESS (clk) BEGIN
        IF rising_edge(clk) THEN
            IF cnt = DIV - 1 THEN
                cnt <= 0;
                ce <= '1';
            ELSE
                cnt <= cnt + 1;
                ce <= '0';
            END IF;
        END IF;
    END PROCESS;
END ARCHITECTURE;