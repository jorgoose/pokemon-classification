from pathlib import Path


# Create a path object
# File path was changed multiple times to organize the files
images = Path("pokemon").rglob('*.png')

# Create a new folder called organized
Path("organized").mkdir(parents=True, exist_ok=True)

iterator = 0

# Iterate through all files in the pokemon folder
for file in images:
    # Check if the image ends with png
    if file.suffix == ".png":
        # Get the number from the filename
        number = file.stem
        # Create a new folder for each number
        Path(f"organized/{number}").mkdir(parents=True, exist_ok=True)

        key = file.stem + str(iterator)

        # Move the file to the new folder with a new name
        file.rename(f"organized/{number}/{key}-{file.stem}{file.suffix}")

        iterator += 1

print('Done')
