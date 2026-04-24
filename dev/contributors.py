#!/usr/bin/env python3

import subprocess
import sys
import re


def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()




def main():
    if len(sys.argv) < 3:
        print('Example Usage: contributors.py "2026-01-01" "2026-03-03"  [--debug]')
        sys.exit(1)

    since, until = sys.argv[1], sys.argv[2]
    debug = "--debug" in sys.argv
    # --- Contributors in range ---
    authors = run(
        f'git log --since="{since}" --until="{until}" --format="%aN <%aE>"'
    ).splitlines()

    # --- Keep longest name for each email?
    email_to_name = {}

    for entry in authors:
        if "[bot]" in entry:
            continue

        if "<" in entry and ">" in entry:
            name, email = entry.rsplit("<", 1)
            name = name.strip()
            email = email.strip(">").strip()
        else:
            continue

        if email not in email_to_name or len(name) > len(email_to_name[email]):
            email_to_name[email] = name

    # --- Identify NEW contributors ---
    new_emails = set()

    for email in email_to_name:
        prev = run(
            f'git log --before="{since}" --author="{email}" -n 1 --format="%H"'
        )

        if not prev:
            new_emails.add(email)

    # --- Sort by last name ---
    def last_name_key(name):
        return name.split()[-1].lower()

    contributors = sorted(
        email_to_name.items(),
        key=lambda item: last_name_key(item[1])
    )

    # --- Summaries ---
    messages = run(
        f'git log --since="{since}" --until="{until}" --format="%s"'
    ).splitlines()

    pr_numbers = set()
    for msg in messages:
        pr_numbers.update(re.findall(r"#(\d+)", msg))

    print("## Contributors\n")

    for email, name in contributors:
        prefix = "-* " if email in new_emails else "- "
        if debug:
            print(f"{prefix}{name} <{email}>")
        else:
            print(f"{prefix}{name}")


    print("")
    print(f"Total contributors: {len(contributors)}")
    print(f"New contributors: {len(new_emails)}")
    print(f"Total merged PRs: {len(pr_numbers)}")

if __name__ == "__main__":
    main()
