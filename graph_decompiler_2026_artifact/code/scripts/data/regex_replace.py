import re
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Find and Replace text in a file using Regex.")
    parser.add_argument("file", help="Path to the file")
    parser.add_argument("pattern", help="The regex pattern to find")
    parser.add_argument("replacement", help="The text to insert in place of the match")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without modifying file")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Perform the replacement
        # re.sub(pattern, repl, string)
        new_content = re.sub(args.pattern, args.replacement, content)

        if args.dry_run:
            print(f"--- Dry Run: Replacing '{args.pattern}' with '{args.replacement}' ---")
            # Print a snippet of the result
            print(new_content[:1000]) 
            print("\n--- End Dry Run ---")
        else:
            with open(args.file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Success: Replaced occurrences of '{args.pattern}' in '{args.file}'.")

    except re.error as e:
        print(f"Regex Error: {e}")

if __name__ == "__main__":
    main()