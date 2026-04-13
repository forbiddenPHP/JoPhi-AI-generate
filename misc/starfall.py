#!/usr/bin/env python3
"""Stars falling through a terminal. No reason. Just because."""

import os, sys, time, random, math

def run():
    cols, rows = os.get_terminal_size()
    stars = []
    t = 0

    print("\033[?25l\033[2J", end="")  # hide cursor, clear
    try:
        while True:
            # spawn
            if random.random() < 0.4:
                stars.append([random.randint(0, cols-1), 0, random.uniform(0.3, 1.0)])

            # move
            buf = [[" "] * cols for _ in range(rows)]
            alive = []
            for s in stars:
                s[1] += s[2]
                y = int(s[1])
                if y < rows:
                    c = "." if s[2] < 0.5 else "+" if s[2] < 0.8 else "*"
                    buf[y][s[0]] = c
                    alive.append(s)
            stars = alive

            # draw
            sys.stdout.write("\033[H")
            for row in buf:
                sys.stdout.write("".join(row) + "\n")
            sys.stdout.flush()

            t += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\033[?25h\033[2J", end="")  # show cursor, clear

if __name__ == "__main__":
    run()
