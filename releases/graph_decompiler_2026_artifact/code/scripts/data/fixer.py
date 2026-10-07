import json
import sys

FIELD_NAME = "language"   # change to "lang" if needed
FIELD_VALUE = "Dart"


def add_language_field(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line_number, line in enumerate(infile, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_number}: {e}", file=sys.stderr)
                continue

            if FIELD_NAME not in obj:
                obj[FIELD_NAME] = FIELD_VALUE

            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_language.py input.jsonl output.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    add_language_field(input_file, output_file)
    print(f"Done. Updated JSONL saved to {output_file}")