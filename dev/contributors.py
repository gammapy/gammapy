#!/usr/bin/env python3
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict


def run(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout


def git(args):
    return run(["git"] + args)


def normalise(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if c.isalnum())


def pick_name(a, b):
    long, short = (a, b) if len(a) >= len(b) else (b, a)
    return long if any(c.isupper() for c in long) else short


def add(contributors, emails, counts, name, email, kind):
    name = name.strip()
    key = email.strip().lower()
    key = key.split("@")[0]

    if "bot" in name.lower() or email.endswith("github.com"):
        return

    if key in contributors:
        name = pick_name(name, contributors[key])

    contributors[key] = name
    emails[key].add(email)

    counts[key][kind] += 1


def is_first_time_contributor(email, since):
    out = git(
        [
            "log",
            f"--until={since}",
            "--author",
            email,
            "-n",
            "1",
            "--pretty=%H",
        ]
    )
    return out.strip() == ""


def main():
    if len(sys.argv) < 3:
        sys.exit('Usage: contributors.py "since" "until" [--min-contributions N]')

    since, until = sys.argv[1], sys.argv[2]

    min_contrib = 0
    if "--min-contributions" in sys.argv:
        i = sys.argv.index("--min-contributions")
        min_contrib = int(sys.argv[i + 1])

    contributors = {}
    emails = defaultdict(set)
    counts = defaultdict(lambda: {"commits": 0, "co": 0, "reviews": 0})

    co = re.compile(r"Co-authored-by:\s*(.+?)\s*<(.+?)>", re.I)
    signed = re.compile(r"Signed-off-by:\s*(.+?)\s*<(.+?)>", re.I)

    log = git(
        [
            "log",
            f"--since={since}",
            f"--until={until}",
            "--format=%H%x01%aN <%aE>%x01%B%x01END",
        ]
    )

    for block in log.split("END"):
        if not block.strip():
            continue

        try:
            _, author, body = block.split("\x01", 2)
        except ValueError:
            continue

        is_signed = False
        for name, email in signed.findall(body):
            add(contributors, emails, counts, name, email, "commits")
            is_signed = True

        if not is_signed and "<" in author:
            name, email = author.rsplit("<", 1)
            add(contributors, emails, counts, name, email.strip(">"), "commits")

        for name, email in co.findall(body):
            add(contributors, emails, counts, name, email, "co")

    msgs = git(
        [
            "log",
            f"--since={since}",
            f"--until={until}",
            "--format=%s",
        ]
    ).splitlines()

    prs = set()
    for msg in msgs:
        for a, b in re.findall(r"(?:\(#(\d+)\)|Merge pull request #(\d+))", msg):
            prs.update(filter(None, (a, b)))

    # detect first-time contributors per key
    is_new_key = {}
    for key, key_emails in emails.items():
        # must be new across ALL known emails
        is_new_key[key] = all(
            is_first_time_contributor(email, since) for email in key_emails
        )

    # merging
    merged = {}
    merged_counts = defaultdict(lambda: {"commits": 0, "co": 0})
    merged_new = defaultdict(lambda: True)

    for key, name in contributors.items():
        mkey = normalise(name)

        if mkey not in merged or len(name) > len(merged[mkey]):
            merged[mkey] = name

        for k in ("commits", "co"):
            merged_counts[mkey][k] += counts[key][k]

        merged_new[mkey] = merged_new[mkey] and is_new_key[key]

    items = [(k, merged[k], merged_counts[k]) for k in merged]

    print("## Contributors\n")
    visible = 0
    new_visible = 0
    for key, name, s in sorted(items, key=lambda x: x[1].split()[-1]):
        total = s["commits"] + s["co"]

        if total < min_contrib:
            continue

        parts = []
        if s["commits"]:
            parts.append(f"{s['commits']} commit{'s' if s['commits'] > 1 else ''}")
        if s["co"]:
            parts.append(f"{s['co']} co-authored")

        star = " "
        if merged_new[key]:
            star = "* "
            new_visible += 1

        print(f"-{star}{name}" + (f" ({', '.join(parts)})" if parts else ""))
        visible += 1

    print()
    print(f"New contributors: {new_visible} (Marked with * in the list)")
    print(f"Total contributors: {visible}")
    print(f"Total merged PRs: {len(prs)}")


if __name__ == "__main__":
    main()
