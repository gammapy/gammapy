#!/usr/bin/env python3
import json
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict


def run(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout


def git(args):
    return run(["git"] + args)


def gh(args):
    return run(["gh"] + args)


def normalise(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if c.isalnum())


def similarity(email, name):
    local = normalise(email.split("@")[0])
    name = normalise(name)
    if not local or not name:
        return 0
    common = sum(1 for c in local if c in name)
    return common / max(len(local), len(name))


def get_github_name(login):
    try:
        data = json.loads(gh(["api", f"users/{login}"]))
        return data.get("name") or login.replace("-", " ").replace("_", " ").title()
    except Exception:
        return login


def get_reviewers(pr):
    data = json.loads(gh(["pr", "view", pr, "--json", "reviews"]))
    return {r["author"]["login"] for r in data.get("reviews", []) if r.get("author")}
    try:
        pass
    except Exception:
        return set()


def match_email_login(email, login):
    local = email.split("@")[0]
    return local in login or login in local


def add(contributors, counts, name, email, kind):
    name = name.strip()
    email = email.strip().lower()

    if "bot" in name.lower() or email.endswith("github.com"):
        return

    contributors[email] = name

    counts[email][kind] += 1


def main():
    if len(sys.argv) < 3:
        sys.exit(
            'Usage: contributors.py "since" "until" '
            "[--include-reviewers] [--min-contributions N] "
            "[--merge-threshold T]"
        )

    since, until = sys.argv[1], sys.argv[2]
    include_reviewers = "--include-reviewers" in sys.argv

    min_contrib = 0
    if "--min-contributions" in sys.argv:
        i = sys.argv.index("--min-contributions")
        min_contrib = int(sys.argv[i + 1])

    contributors = {}
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
            add(contributors, counts, name, email, "commits")
            is_signed = True

        if not is_signed and "<" in author:
            name, email = author.rsplit("<", 1)
            add(contributors, counts, name, email.strip(">"), "commits")

        for name, email in co.findall(body):
            add(contributors, counts, name, email, "co")

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

    if include_reviewers:
        for pr in prs:
            for login in get_reviewers(pr):
                hit = next(
                    (
                        e
                        for e in contributors
                        if not e.startswith("gh:") and match_email_login(e, login)
                    ),
                    None,
                )

                if hit:
                    counts[hit]["reviews"] += 1
                else:
                    key = f"gh:{login}"
                    contributors[key] = get_github_name(login)
                    counts[key]["reviews"] += 1

    # merging
    merged = {}
    merged_counts = defaultdict(lambda: {"commits": 0, "co": 0, "reviews": 0})

    for email, name in contributors.items():
        key = normalise(name)

        if key not in merged or len(name) > len(merged[key]):
            merged[key] = name

        for k in ("commits", "co", "reviews"):
            merged_counts[key][k] += counts[email][k]

    items = [(merged[k], merged_counts[k]) for k in merged]

    print("## Contributors\n")

    visible = 0

    for name, s in sorted(items, key=lambda x: x[0].split()[-1]):
        total = s["commits"] + s["co"] + s["reviews"]

        if total < min_contrib:
            continue

        parts = []
        if s["commits"]:
            parts.append(f"{s['commits']} commit{'s' if s['commits'] > 1 else ''}")
        if s["co"]:
            parts.append(f"{s['co']} co-authored")
        if s["reviews"]:
            parts.append(f"{s['reviews']} review{'s' if s['reviews'] > 1 else ''}")

        print(f"- {name}" + (f" ({', '.join(parts)})" if parts else ""))
        visible += 1

    print()
    print(f"Total contributors: {visible}")
    print(f"Total merged PRs: {len(prs)}")


if __name__ == "__main__":
    main()
