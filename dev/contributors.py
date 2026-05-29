#!/usr/bin/env python3
import subprocess
import sys
import re


def run_git(args):
    return subprocess.run(
        ["git"] + args,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def last_name_key(name):
    return name.split()[-1].lower()


def main():
    if len(sys.argv) < 3:
        print('Example Usage: contributors.py "2026-01-01" "2026-03-03" [--debug]')
        sys.exit(1)

    since, until = sys.argv[1], sys.argv[2]
    debug = "--debug" in sys.argv

    authors = run_git(
        ["log", f"--since={since}", f"--until={until}", "--format=%aN <%aE>"]
    ).splitlines()

    email_to_name = {}

    for entry in authors:
        if "<" in entry and ">" in entry:
            name, email = entry.rsplit("<", 1)
            name = name.strip()
            email = email.strip(">").strip()
        else:
            continue

        if (
            "[bot]" in entry
            or email.endswith("@users.noreply.github.com")
            or email.endswith("@github.com")
        ):
            continue

        if email not in email_to_name or len(name) > len(email_to_name[email]):
            email_to_name[email] = name

    name_to_emails = {}
    for email, name in email_to_name.items():
        name_to_emails.setdefault(name, []).append(email)

    new_names = set()

    for name, emails in name_to_emails.items():
        is_new = True
        for email in emails:
            prev = run_git(
                [
                    "log",
                    f"--before={since}",
                    f"--author=<{email}>",
                    "-n",
                    "1",
                    "--format=%H",
                ]
            )
            if prev:
                is_new = False
                break
        if is_new:
            new_names.add(name)

    contributors = sorted(email_to_name.items(), key=lambda x: last_name_key(x[1]))

    messages = run_git(
        ["log", f"--since={since}", f"--until={until}", "--format=%s"]
    ).splitlines()

    pattern = r"(?:\(#(\d+)\)|Merge pull request #(\d+))"
    pr_numbers = set()

    for msg in messages:
        for m in re.findall(pattern, msg):
            pr_numbers.update(filter(None, m))

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
    print(f"New contributors: {len(new_names)} (Marked with * in the list)")
    print(f"Total merged PRs: {len(pr_numbers)}")

    if debug:
        print("\n## Debug: PR list\n")
        for msg in messages:
            if re.search(pattern, msg):
                print(msg)


if __name__ == "__main__":
    main()
