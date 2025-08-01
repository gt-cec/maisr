#!/usr/bin/env python3
"""
Script to parse cProfile output and sort by tottime (total time).
"""

import re
import sys
from typing import List, Tuple, NamedTuple


class ProfileEntry(NamedTuple):
    ncalls: str
    tottime: float
    percall_tot: float
    cumtime: float
    percall_cum: float
    filename_function: str
    original_line: str


def parse_profile_line(line: str) -> ProfileEntry:
    """
    Parse a single line from cProfile output.
    Expected format: ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    """
    # Remove leading/trailing whitespace
    line = line.strip()

    # Split by whitespace, but be careful with the filename part
    parts = line.split()

    if len(parts) < 6:
        raise ValueError(f"Invalid line format: {line}")

    # The filename:lineno(function) part might contain spaces, so join the last parts
    ncalls = parts[0]
    tottime = float(parts[1])
    percall_tot = float(parts[2])
    cumtime = float(parts[3])
    percall_cum = float(parts[4])
    filename_function = ' '.join(parts[5:])

    return ProfileEntry(
        ncalls=ncalls,
        tottime=tottime,
        percall_tot=percall_tot,
        cumtime=cumtime,
        percall_cum=percall_cum,
        filename_function=filename_function,
        original_line=line
    )


def parse_profile_file(filename: str) -> List[ProfileEntry]:
    """
    Parse a cProfile output file and return a list of ProfileEntry objects.
    """
    entries = []

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and header lines
            if not line or line.startswith('ncalls') or 'Random listing order' in line:
                continue

            # Skip lines that don't look like profile entries
            # Profile entries should start with numbers
            if not re.match(r'^\s*\d', line):
                continue

            try:
                entry = parse_profile_line(line)
                entries.append(entry)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {line_num}: {line}", file=sys.stderr)
                print(f"Error: {e}", file=sys.stderr)
                continue

    return entries


def format_profile_entry(entry: ProfileEntry) -> str:
    """
    Format a ProfileEntry back to the original cProfile format.
    """
    return f"{entry.ncalls:>8} {entry.tottime:>8.3f} {entry.percall_tot:>8.3f} {entry.cumtime:>8.3f} {entry.percall_cum:>8.3f} {entry.filename_function}"


def main():
    if len(sys.argv) not in [2, 3]:
        print("Usage: python profile_parser.py <profile_file> [output_file]")
        print("This script parses cProfile output and sorts by tottime (descending)")
        print("If output_file is not specified, uses <profile_file>_sorted.txt")
        sys.exit(1)

    filename = sys.argv[1]

    # Determine output filename
    if len(sys.argv) == 3:
        output_filename = sys.argv[2]
    else:
        # Create output filename by adding '_sorted' before the extension
        if '.' in filename:
            name, ext = filename.rsplit('.', 1)
            output_filename = f"{name}_sorted.{ext}"
        else:
            output_filename = f"{filename}_sorted.txt"

    try:
        # Parse the profile file
        entries = parse_profile_file(filename)

        if not entries:
            print("No valid profile entries found in the file.")
            sys.exit(1)

        # Sort by tottime in descending order (highest first)
        sorted_entries = sorted(entries, key=lambda x: x.tottime, reverse=True)

        # Write to output file
        with open(output_filename, 'w') as f:
            # Write header
            f.write(
                f"{'ncalls':>8} {'tottime':>8} {'percall':>8} {'cumtime':>8} {'percall':>8} filename:lineno(function)\n")
            f.write("-" * 80 + "\n")

            # Write sorted entries
            for entry in sorted_entries:
                f.write(format_profile_entry(entry) + "\n")

            # Write summary
            f.write(f"\nTotal entries processed: {len(entries)}\n")
            f.write(
                f"Top function by tottime: {sorted_entries[0].filename_function} ({sorted_entries[0].tottime:.3f}s)\n")

        print(f"Successfully processed {len(entries)} entries from '{filename}'")
        print(f"Sorted output saved to '{output_filename}'")
        print(f"Top function by tottime: {sorted_entries[0].filename_function} ({sorted_entries[0].tottime:.3f}s)")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied writing to '{output_filename}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()