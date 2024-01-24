#!/usr/bin/env python3

# Kills all processes whos command line match provided pattern

import os
import psutil
import signal
import sys


def main():
    if len(sys.argv) == 1:
        sys.stderr.write("Error: no search pattern provided")
        sys.exit(1)

    search_pattern = sys.argv[1]

    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'clang' in process.info['name']:

            PID = []
            if search_pattern in ' '.join(process.info['cmdline']):
                PID.append(process.info['pid'])

            if len(PID) == 0:
                return
            
            if len(PID) > 1:
                sys.stderr.write("Error: more then one process matches search pattern")

            os.kill(PID[0], signal.SIGTERM)


if __name__ == "__main__":
    main()
