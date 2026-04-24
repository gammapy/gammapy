#!/usr/bin/env python3

import subprocess
import sys
import re

def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()

def last_name_key(name):
    # --- Sort by last name ---
    return name.split()[-1].lower()

def main():
    if len(sys.argv) < 3:
        print('Example Usage: contributors.py "2026-01-01" "2026-03-03"  [--debug]')
        sys.exit(1)

    since, until = sys.argv[1], sys.argv[2]
    debug = "--debug" in sys.argv


    authors = run(
        f'git log --since="{since}" --until="{until}" --format="%aN <%aE>"'
    ).splitlines()

    ## Now check for uniqueness of email, then of name

    # (a) Keep unique emails
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

    # (b) Unique names
    name_to_emails = {}
    for email, name in email_to_name.items():
        name_to_emails.setdefault(name, []).append(email)

    # --- Identify new contributors by email id---
    new_names = set()

    for name, emails in name_to_emails.items():
        is_new = True
        for email in emails:
            prev = run( f'git log --before="{since}" --author="<{email}>" -n 1 --format="%H"' )
            if prev:
                is_new = False
                break
        if is_new:
            new_names.add(name)

    contributors = sorted(
        email_to_name.items(),
        key=lambda item: last_name_key(item[1])
    )

    # --- Summaries ---
    messages = run(
        f'git log --since="{since}" --until="{until}" --format="%s"'
    ).splitlines()

    pattern = r"(?:\(#(\d+)\)|Merge pull request #(\d+))"
    pr_numbers = set()
    for msg in messages:
        pr_numbers.update(re.findall(pattern, msg))

    print("## Contributors\n")
    seen = set()
    for email, name in contributors:
        if name in seen:
            continue
        seen.add(name)
        prefix = "-* " if name in new_names else "- "
        if debug:
            print(f"{prefix}{name} <{email}>")
        else:
            print(f"{prefix}{name}")


    print("")
    print(f"Total contributors: {len(contributors)} (See below for full list)")
    print(f"New contributors: {len(new_names)}  (Marked with * in the list)")
    print(f"Total merged PRs: {len(pr_numbers)}")

    print("\n\n")

    if debug:
        print("## Debug: PR list (full messages)\n")
        for msg in messages:
            matches = re.findall(pattern, msg)
            for pr in matches:
                print(f"- PR #{pr}", msg)

if __name__ == "__main__":
    main()
