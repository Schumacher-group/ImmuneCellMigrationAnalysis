#!/bin/bash

# Specify the source and destination prefixes
source_prefix="MCR_data"
destination_prefix="Control_data"

# Iterate over files matching the source prefix
for file in "${source_prefix}"*.npy; do
    # Check if the file exists
    if [ -e "$file" ]; then
        # Extract the filename without the path
        filename=$(basename "$file")

        # Construct the new filename with the destination prefix
        new_filename="${destination_prefix}${filename#${source_prefix}}"

        # Rename the file
        mv "$file" "$new_filename"

        echo "Renamed: $file -> $new_filename"
    fi
done

