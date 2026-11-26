"""Identify a NASA deployment panel independently of predictions and CSV row order."""

import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal


def _digest(rows):
    data = json.dumps(sorted(rows), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _number(value):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Non-finite dose or time")
    return format(number.normalize(), "f") if number else "0"


def cohort_identity(rows):
    """Hash nucleus-to-bag assignments and exposed/sham pairings in an identified panel."""
    members = []
    samples = set()
    nuclei = set()
    bags = {}
    counts = Counter()
    for row in rows:
        sample, nucleus, bag, source, strain, sex, branch = [
            row[field] for field in ("sample_id", "nucleus_key", "sample_name", "source_name",
                                     "strain", "sex", "particle_type")
        ]
        if not all(isinstance(v, str) and v.strip() == v and v
                   for v in (sample, nucleus, bag, source, strain, sex, branch)):
            raise ValueError("Missing or ambiguous cohort identity")
        if sample in samples or nucleus in nuclei:
            raise ValueError("Duplicate sample or nucleus")
        samples.add(sample)
        nuclei.add(nucleus)
        dose = _number(row["dose_Gy"])
        hour = _number(row["hr_post_exposure"])
        identity = (source, strain, sex, branch, dose, hour)
        if bags.setdefault(bag, identity) != identity:
            raise ValueError("Conflicting bag membership")
        counts[bag] += 1
        members.append((sample, nucleus, bag, *identity))
    pairs = defaultdict(dict)
    doses = {"Fe": "0.82", "X-ray": "1"}
    for bag, (source, _, _, branch, dose, hour) in bags.items():
        if branch not in doses or dose not in ("0", doses[branch]) or hour not in ("4", "24", "48"):
            raise ValueError("Unrecognized radiation condition")
        pair = pairs[(source, branch, hour)]
        if dose in pair:
            raise ValueError("Ambiguous radiation contrast")
        pair[dose] = bag
    contrasts = []
    for (source, branch, hour), pair in pairs.items():
        if set(pair) != {"0", doses[branch]}:
            raise ValueError("Missing sham or exposed bag")
        contrasts.append((source, branch, hour, doses[branch], pair["0"], pair[doses[branch]]))
    if len(members) != 2400 or len(bags) != 24 or set(counts.values()) != {100} or len(contrasts) != 12:
        raise ValueError("Incomplete development panel")
    return {
        "cohort_membership_sha256": _digest(members),
        "contrast_membership_sha256": _digest(contrasts),
        "nuclei": len(members), "bags": len(bags), "contrasts": len(contrasts),
        "sources": sorted({identity[0] for identity in bags.values()}),
    }


def require_cohort(rows, policy):
    result = cohort_identity(rows)
    for key, value in result.items():
        if policy.get(key) != value:
            raise ValueError("Development panel differs from the frozen policy: " + key)
    return result
