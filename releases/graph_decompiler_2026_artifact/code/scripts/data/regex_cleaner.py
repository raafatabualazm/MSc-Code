import re
import argparse
import sys
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Remove text matching a Regex from a file.")
    parser.add_argument("file", help="Path to the file you want to clean")
    parser.add_argument("regex", help="The regular expression pattern to remove")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without modifying the file")
    
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Perform the substitution
        new_content = re.sub(args.regex, '', content)

        if args.dry_run:
            print("--- Dry Run Output (First 500 chars) ---")
            print(new_content[:500])
            print("\n--- End Dry Run ---")
            print(f"Original Length: {len(content)} | New Length: {len(new_content)}")
        else:
            with open(args.file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Success: Removed matches of '{args.regex}' from '{args.file}'.")

    except re.error as e:
        print(f"Regex Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()