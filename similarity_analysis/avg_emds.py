import re


def calculate_average_from_file(filename):
    """
    Read a file and calculate the average of all numerical values.

    Args:
        filename (str): Path to the input file

    Returns:
        float: Average of all numbers in the file
    """
    numbers = []

    try:
        with open(filename, 'r') as file:
            content = file.read()

        # Extract all floating point numbers from the file
        # This regex matches numbers with decimal points
        number_pattern = r'\d+\.\d+'
        matches = re.findall(number_pattern, content)

        # Convert string matches to float
        numbers = [float(match) for match in matches]

        if not numbers:
            print("No numbers found in the file.")
            return None

        # Calculate average
        average = sum(numbers) / len(numbers)

        print(f"Found {len(numbers)} numbers")
        print(f"Sum: {sum(numbers):.6f}")
        print(f"Average: {average:.6f}")

        return average

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_average_from_string(data_string):
    """
    Calculate average from a string containing the data.
    Useful if you want to paste the data directly into the script.
    """
    numbers = []

    # Extract all floating point numbers from the string
    number_pattern = r'\d+\.\d+'
    matches = re.findall(number_pattern, data_string)

    # Convert string matches to float
    numbers = [float(match) for match in matches]

    if not numbers:
        print("No numbers found in the data.")
        return None

    # Calculate average
    average = sum(numbers) / len(numbers)

    print(f"Found {len(numbers)} numbers")
    print(f"Sum: {sum(numbers):.6f}")
    print(f"Average: {average:.6f}")

    return average


# Example usage:
if __name__ == "__main__":
    # Method 1: Read from file
    filename = "emds.txt"
    print("Calculating average from file:")
    calculate_average_from_file(filename)

    print("\n" + "=" * 50 + "\n")