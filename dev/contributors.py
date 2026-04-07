#!/usr/bin/env python3

import subprocess
import sys
import re


def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def main():
    if len(sys.argv) != 3:
        print('Example Usage: contributors.py "2026-01-01" "2026-03-03"')
        sys.exit(1)

    since, until = sys.argv[1], sys.argv[2]

    # --- Contributors (names) ---
    authors = run(
        f'git log --since="{since}" --until="{until}" --format="%aN <%aE>"'
    ).splitlines()

    email_to_name = {}

    for author in authors:
        if "[bot]" in author:
            continue

        name, email = author.rsplit("<", 1)
        name = name.strip()
        email = email.strip(">").strip()

        # Keep the longest name for each email?
        if email not in email_to_name or len(name) > len(email_to_name[email]):
            email_to_name[email] = name

    contributors = sorted(set(email_to_name.values()))

    # --- Commit messages (for PR extraction) ---
    messages = run(
        f'git log --since="{since}" --until="{until}" --format="%s"'
    ).splitlines()

    pr_numbers = set()

    for msg in messages:
        matches = re.findall(r"#(\d+)", msg)
        pr_numbers.update(matches)

    # --- Output ---
    print("## Contributors\n")

    for name in contributors:
        print(f"- {name}")

    print("")
    print(f"**Total merged PRs:** {len(pr_numbers)}")


if __name__ == "__main__":
    main()
