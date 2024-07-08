import argparse

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Add one to an integer.")
    parser.add_argument('integer', type=int, help='An integer to which 1 will be added')
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    result = args.integer + 1
    print(f'The result is: {result}')

if __name__ == '__main__':
    main()

