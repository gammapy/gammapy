#!/usr/bin/env python3

import subprocess
import sys
import re

def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def main():
    if len(sys.argv) < 3:
        print('Example Usage: contributors.py "2026-01-01" "2026-03-03"  [--email]')
        sys.exit(1)

    since, until = sys.argv[1], sys.argv[2]
    print_email = "--email" in sys.argv
    # --- Contributors in range ---
    authors = run(
        f'git log --since="{since}" --until="{until}" --format="%aN <%aE>"'
    ).splitlines()

    # Keep the longest name for each email?
    email_to_name = {}

    for entry in authors:
        if "<" in entry and ">" in entry:
            name, email = entry.rsplit("<", 1)
            name = name.strip()
            email = email.strip(">").strip()
        else:
            continue

        if "[bot]" in entry or email.endswith("@users.noreply.github.com") or email.endswith("@github.com"):
            continue

        if email not in email_to_name or len(name) > len(email_to_name[email]):
            email_to_name[email] = name

    # --- Identify new contributors by email id---
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
    unique_names = set()
    for email, name in contributors:
        if name in unique_names:
            continue
        unique_names.add(name)
        prefix = "-* " if email in new_emails else "- "
        if print_email:
            print(f"{prefix}{name} <{email}>")
        else:
            print(f"{prefix}{name}")


    print("")
    print(f"Total contributors: {len(contributors)}")
    print(f"New contributors: {len(new_emails)}")
    print(f"Total merged PRs: {len(pr_numbers)}")

if __name__ == "__main__":
    main()
