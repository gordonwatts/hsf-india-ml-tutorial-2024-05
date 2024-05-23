import argparse
from pathlib import Path

import uproot


def split_root_file(input_file, output_file):
    # Load the root file using uproot
    root_file = uproot.open(input_file)

    # Get the name of the tree in the root file
    tree_name = root_file.keys()[0]

    # Get the number of events in the tree
    num_events = root_file[tree_name].num_entries  # type: ignore

    # Calculate the range of events to keep and remove
    keep_range = (0, num_events - 10000)
    remove_range = (num_events - 10000, num_events)

    # Open the output file for writing
    output = uproot.recreate(output_file)

    # Copy all branches and events except the last 10000
    output[tree_name] = root_file[tree_name].arrays(  # type: ignore
        entry_start=keep_range[0], entry_stop=keep_range[1]
    )
    output.close()

    # Create a second output file for the removed range
    removed_output_file = output_file.with_name(
        output_file.stem + "_removed" + output_file.suffix
    )

    # Copy the last 10000 events to the second output file
    removed_output = uproot.recreate(removed_output_file)
    removed_output[tree_name] = root_file[tree_name].arrays(  # type: ignore
        entry_start=remove_range[0], entry_stop=remove_range[1]
    )
    removed_output.close()

    print(f"Splitting complete. Saved {keep_range[1]} events to {output_file}.")
    print(f"Saved {remove_range[1] - remove_range[0]} events to {removed_output_file}.")


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Split a ROOT file")

    # Add the input file argument
    parser.add_argument("input_file", type=Path, help="Input ROOT file")

    # Add the output file argument
    parser.add_argument("output_file", type=Path, help="Output ROOT file")

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the split_root_file function with the provided file names
    split_root_file(args.input_file, args.output_file)
