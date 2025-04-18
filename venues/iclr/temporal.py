import os
import json
from collections import defaultdict
from typing import List, Dict
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import csv

FIELDS = ["confidence", "correctness", "technical_novelty"]

def parse_day(day):
    return list(zip(
        map(int, day["rating"].split(';')),
        map(int, day["confidence"].split(';')),
        map(int, day["correctness"].split(';')),
        map(int, day["technical_novelty"].split(';'))
    ))

def signature_cost(sig1, sig2):
    return sum(abs(a - b) for a, b in zip(sig1, sig2))

def trace_with_hungarian(data, max_cost_threshold=3):
    
    """
    Tracks reviewer identities backward in time using dynamic programming
    (Hungarian algorithm) to minimize profile differences between consecutive days.

    This function assumes that the list of reviews on each day is sorted by "rating"
    in ascending order, and that this sorting may shuffle the position of reviewers
    if their ratings change from day to day.

    The canonical reviewer ordering is taken from the last day (latest snapshot),
    where each index (0 to N-1) is considered the true identity of a reviewer.
    The goal is to trace back these canonical reviewer IDs through earlier days,
    despite possible shuffling in their positions due to score updates.

    Matching is based on the similarity of the reviewer signature:
        (confidence, correctness, technical_novelty)

    For each earlier day:
        - A cost matrix is computed between all reviewers on that day and all
          canonical reviewers from the following day.
        - The cost is defined as the L1 distance (sum of absolute differences)
          between reviewer signatures.
        - The Hungarian algorithm is used to find the optimal 1-to-1 assignment
          (minimum total cost).
        - If a match has a cost greater than `max_cost_threshold`, the match is
          considered unreliable and that reviewer is marked with `-1`.

    Parameters:
    ----------
    data : List[Dict]
        A list of daily review snapshots (sorted by time), where each entry is a dict
        containing:
            - 'rating': semicolon-separated scores (e.g., "3;5;6;6;6")
            - 'confidence': semicolon-separated values
            - 'correctness': semicolon-separated values
            - 'technical_novelty': semicolon-separated values
            - 'time_code': a date string (e.g., "11202023")

    max_cost_threshold : int, optional (default=3)
        The maximum allowable distance between reviewer profiles to consider a match valid.
        If the computed cost exceeds this threshold, the reviewer is assumed to be
        unmatchable and marked as -1 in the output.

    Returns:
    -------
    List[Dict[str, List[int]]]
        A list of dictionaries, one per day, of the form:
            [{time_code: [canonical_reviewer_ids]}]
        Where each list maps the index of the reviewer on that day to their canonical
        reviewer ID (based on the last day). If a reviewer cannot be reliably matched,
        the ID will be -1.
    """
    
    n_days = len(data)
    n_reviewers = len(data[-1]["rating"].split(';'))

    # Canonical: last day
    result = [{} for _ in range(n_days)]
    result[-1] = {data[-1]['time_code']: list(range(n_reviewers))}
    canonical_signatures = [x[1:] for x in parse_day(data[-1])]  # ignore rating

    for day_idx in range(n_days - 2, -1, -1):
        day_sigs = [x[1:] for x in parse_day(data[day_idx])]  # skip rating

        cost_matrix = np.zeros((n_reviewers, n_reviewers), dtype=int)
        for i in range(n_reviewers):
            for j in range(n_reviewers):
                cost_matrix[i, j] = signature_cost(day_sigs[i], canonical_signatures[j])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Construct match list
        matched = [-1] * n_reviewers
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= max_cost_threshold:
                matched[i] = j
            else:
                matched[i] = -1

        result[day_idx] = {data[day_idx]['time_code']: matched}

    return result


def sort_key(entry):
    mmddyyyy = entry['time_code']
    return mmddyyyy[4:] + mmddyyyy[:4]


def analyze_stability(results: Dict[str, List[Dict]], mode: str = "first_last") -> Dict[str, Dict[str, Dict[str, float]]]:
    total_papers = 0
    total_reviewers = 0
    unchanged = {
        "confidence": {"papers": 0, "reviewers": 0},
        "correctness": {"papers": 0, "reviewers": 0},
        "technical_novelty": {"papers": 0, "reviewers": 0},
        "all_three": {"papers": 0, "reviewers": 0}
    }
    metadata = {
        "fully_empty_profiles": 0,
        "papers_with_extra_reviewers": 0
    }

    failed_records = {}

    for paper_id, entries in results.items():
        entries.sort(key=sort_key)
        if len(entries) <= 1:
            continue

        initial_raw = entries[0]
        final_raw = entries[-1]

        if any(initial_raw[field].strip() == '' for field in FIELDS):
            metadata["fully_empty_profiles"] += 1
            continue

        reviewer_counts = [len(entry["rating"].split(';')) for entry in entries]
        if any(c != reviewer_counts[0] for c in reviewer_counts):
            metadata["papers_with_extra_reviewers"] += 1
            continue

        reviewer_count = reviewer_counts[0]
        total_papers += 1
        total_reviewers += reviewer_count

        reviewer_flags = [[True, True, True] for _ in range(reviewer_count)]
        paper_flags = {"confidence": True, "correctness": True, "technical_novelty": True}

        if mode == "first_last":
            initial = parse_day(initial_raw)
            final = parse_day(final_raw)
            for i in range(reviewer_count):
                if initial[i][1] != final[i][1]:
                    reviewer_flags[i][0] = False
                    paper_flags["confidence"] = False
                if initial[i][2] != final[i][2]:
                    reviewer_flags[i][1] = False
                    paper_flags["correctness"] = False
                if initial[i][3] != final[i][3]:
                    reviewer_flags[i][2] = False
                    paper_flags["technical_novelty"] = False
        else:
            for i in range(1, len(entries)):
                prev = parse_day(entries[i - 1])
                curr = parse_day(entries[i])
                for j in range(reviewer_count):
                    if prev[j][1] != curr[j][1]:
                        reviewer_flags[j][0] = False
                        paper_flags["confidence"] = False
                    if prev[j][2] != curr[j][2]:
                        reviewer_flags[j][1] = False
                        paper_flags["correctness"] = False
                    if prev[j][3] != curr[j][3]:
                        reviewer_flags[j][2] = False
                        paper_flags["technical_novelty"] = False

        for flag in reviewer_flags:
            if flag[0]:
                unchanged["confidence"]["reviewers"] += 1
            if flag[1]:
                unchanged["correctness"]["reviewers"] += 1
            if flag[2]:
                unchanged["technical_novelty"]["reviewers"] += 1
            if all(flag):
                unchanged["all_three"]["reviewers"] += 1

        for key in FIELDS:
            if paper_flags[key]:
                unchanged[key]["papers"] += 1

        if all(paper_flags.values()):
            unchanged["all_three"]["papers"] += 1
        else:
            failed_records[paper_id] = entries

    def percentage(part, whole):
        return round(part / whole * 100, 2) if whole else 0.0

    total_skipped = metadata["fully_empty_profiles"] + metadata["papers_with_extra_reviewers"]
    total_counted = total_papers + total_skipped

    return {
        "mode": mode,
        "paper_level": {
            "total_papers": total_papers,
            "unchanged": {
                key: {
                    "count": unchanged[key]["papers"],
                    "percentage": percentage(unchanged[key]["papers"], total_papers)
                } for key in ["confidence", "correctness", "technical_novelty", "all_three"]
            }
        },
        "reviewer_level": {
            "total_reviewers": total_reviewers,
            "unchanged": {
                key: {
                    "count": unchanged[key]["reviewers"],
                    "percentage": percentage(unchanged[key]["reviewers"], total_reviewers)
                } for key in ["confidence", "correctness", "technical_novelty", "all_three"]
            }
        },
        "metadata": metadata,
        "paper_counts": {
            "valid": total_papers,
            "fully_empty_profiles": metadata["fully_empty_profiles"],
            "extra_reviewers": metadata["papers_with_extra_reviewers"],
            "total_evaluated": total_counted
        },
        "failed_records": failed_records
    }
    
def print_colored(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def print_comparison_table(first, all_days, level):
    headers = ["Dimension", "First & Last", "All Days", "Δ Difference"]
    rows = []
    for key in first[level]["unchanged"]:
        key_label = key.replace('_', ' ').capitalize()
        f_val = first[level]["unchanged"][key]
        a_val = all_days[level]["unchanged"][key]
        f_str = f"{f_val['count']} ({f_val['percentage']}%)"
        a_str = f"{a_val['count']} ({a_val['percentage']}%)"
        diff = round(f_val["percentage"] - a_val["percentage"], 2)
        diff_str = f"{diff:+.2f}%"
        diff_color = "1;32" if diff >= 0 else "1;31"
        rows.append([
            print_colored(key_label, "1;36"),
            print_colored(f_str, "1;33"),
            print_colored(a_str, "1;35"),
            print_colored(diff_str, diff_color)
        ])
    print(print_colored(f"\n📊 Comparison Table — {level.replace('_', ' ').capitalize()}", "1;34"))
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

def print_block(stats):
    mode_str = "First & Last Snapshot" if stats['mode'] == "first_last" else "Across All Days"
    print(print_colored(f"\n=== Stability Analysis ({mode_str}) ===", "1;36"))

    # Introductory explanation
    if stats['mode'] == "first_last":
        print(print_colored("🧠 This mode compares only the first and last snapshots of each paper's review timeline.\n"
                            "If a reviewer's score remained the same from beginning to end, it is considered stable — even if it changed temporarily in between.\n", "1;37"))
    else:
        print(print_colored("🔍 This mode checks reviewer stability across all snapshots in the timeline.\n"
                            "A reviewer's score must remain unchanged at all time points to be considered stable.\n", "1;37"))

    print(print_colored("Paper Counts Breakdown:", "1;33"))
    print(f"  {print_colored('Valid:', '1;33')} {stats['paper_counts']['valid']} papers — included in the stability evaluation")
    print(f"  {print_colored('Fully empty profiles:', '1;33')} {stats['paper_counts']['fully_empty_profiles']} papers — skipped due to missing scores")
    print(f"  {print_colored('Extra reviewers:', '1;33')} {stats['paper_counts']['extra_reviewers']} papers — skipped due to reviewer number inconsistencies")
    print(print_colored(f"\nTotal evaluated papers: {stats['paper_counts']['total_evaluated']}", "1;33"))
    print(print_colored(f"Total reviewers analyzed: {stats['reviewer_level']['total_reviewers']}", "1;33"))

    print(print_colored("\n📘 Paper-Level Stability:", "1;34"))
    print(print_colored("Each paper is marked as stable in a dimension only if all reviewers remained stable in that dimension.", "0;37"))
    for key, values in stats['paper_level']['unchanged'].items():
        print(f"  {print_colored(key.capitalize(), '1;32')}: "
            f"{values['count']} papers ({values['percentage']}%)")

    print(print_colored("\n👤 Reviewer-Level Stability:", "1;34"))
    print(print_colored("Each reviewer is evaluated independently — a reviewer is stable if their score remained unchanged over time.", "0;37"))
    for key, values in stats['reviewer_level']['unchanged'].items():
        print(f"  {print_colored(key.capitalize(), '1;35')}: "
            f"{values['count']} reviewers ({values['percentage']}%)")


def trace_failed_records(
    failed_records: Dict[str, List[Dict]],
    max_cost_threshold: int = 3,
    debug_dir: str = None
) -> Dict[str, List[Dict]]:
    """
    Traces reviewer identities for failed records using the Hungarian matching algorithm,
    only keeping those where all reviewer assignments are successful (no unmatched -1).
    
    Optionally saves debug outputs in CSV format for each traced paper.

    Parameters:
    ----------
    failed_records : Dict[str, List[Dict]]
        Papers that failed the initial stability check.

    max_cost_threshold : int
        Maximum allowed cost between reviewer signatures to accept a match.

    debug_dir : str
        Optional directory to save CSV debug outputs for traced footprints and review entries.

    Returns:
    -------
    traced_entries : Dict[str, List[Dict]]
        Papers for which reviewer identities could be reliably traced.
    """
    traced_entries = {}
    total = len(failed_records)
    success = 0

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    for paper_id, records in tqdm(failed_records.items(), desc="Tracing failed records"):
        try:
            footprints = trace_with_hungarian(records, max_cost_threshold=max_cost_threshold)
            all_days_traced = all(
                -1 not in trace.get(time_code, [])
                for trace in footprints
                for time_code in trace
            )
            if all_days_traced:
                traced_entries[paper_id] = records
                success_flag = True
                success += 1
            else:
                success_flag = False

            # Debug CSV saving
            if debug_dir:
                filename = os.path.join(debug_dir, f"{paper_id}_{'success' if success_flag else 'fail'}.csv")
                with open(filename, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["time_code", "rating", "confidence", "correctness", "technical_novelty", "canonical_ids"])
                    for i, day in enumerate(records):
                        time_code = day["time_code"]
                        trace_map = list(footprints[i].values())[0]
                        writer.writerow([
                            time_code,
                            day["rating"],
                            day["confidence"],
                            day["correctness"],
                            day["technical_novelty"],
                            ";".join(map(str, trace_map))
                        ])

        except Exception as e:
            print(f"⚠️ Error tracing paper {paper_id}: {e}")

    print(print_colored(
        f"\n✅ Successfully traced {success} out of {total} failed papers "
        f"({round(success / total * 100, 2)}%) using max_cost_threshold = {max_cost_threshold}",
        "1;32" if success else "1;31"
    ))
    return traced_entries

    
def full_pipeline(root_folder: str, tracing_threshold: int = 3) -> Dict[str, Dict[str, Dict[str, float]]]:
    id_to_entries = defaultdict(list)

    for filename in os.listdir(root_folder):
        if filename.endswith('.json') and filename.startswith('iclr2024.'):
            time_code = filename.split('.')[1]
            with open(os.path.join(root_folder, filename), 'r') as f:
                data = json.load(f)
                for entry in data:
                    if entry.get('id') and all(field in entry for field in FIELDS):
                        entry['time_code'] = time_code
                        id_to_entries[entry['id']].append(entry)

    id_to_filtered_entries = {}
    for paper_id, entries in id_to_entries.items():
        entries.sort(key=sort_key)
        if len(entries) > 1:
            id_to_filtered_entries[paper_id] = entries

    # Step 1: Run original stability analysis
    stats_first_last = analyze_stability(id_to_filtered_entries, mode="first_last")
    stats_all_days = analyze_stability(id_to_filtered_entries, mode="all_days")

    # Step 2: Print stability analysis immediately
    print_block(stats_first_last)
    print_block(stats_all_days)
    print_comparison_table(stats_first_last, stats_all_days, level="paper_level")
    print_comparison_table(stats_first_last, stats_all_days, level="reviewer_level")

    # Step 3: Extract failed records and trace them
    failed_records = stats_all_days["failed_records"]
    traced_entries = trace_failed_records(failed_records, max_cost_threshold=tracing_threshold, debug_dir=root_folder + f"_footprints_threshold_{tracing_threshold}")

    return {
        "first_last": stats_first_last,
        "all_days": stats_all_days,
        "traced_success": traced_entries
    }

def test(root_folder, target_id):
    
    results = []

    # loop through all files in the root folder
    for filename in os.listdir(root_folder):
        if filename.endswith('.json') and filename.startswith('iclr2024.'):
            # extract the date part from filename (MMDDYYYY)
            time_code = filename.split('.')[1]  

            with open(os.path.join(root_folder, filename), 'r') as f:
                data = json.load(f)

                # check if target_id exists
                # loop through all entries
                for entry in data:
                    if entry.get('id') == target_id:
                        entry['time_code'] = time_code
                        results.append(entry)
    
    # sort by YYYYMMDD (for correct date order)
    def sort_key(entry):
        mmddyyyy = entry['time_code']
        return mmddyyyy[4:] + mmddyyyy[:4]  # YYYYMMDD

    results.sort(key=sort_key)

    # save results
    # with open(output_file, 'w') as f:
        # json.dump(results, f, indent=4)

    # print(f"Saved {len(results)} records to {output_file}")
    
    # footprints = get_reviewer_footprints(results)
    footprints = trace_with_hungarian(results)
    
    return footprints


if __name__ == "__main__":
    
    # loop through the root folder of iclr2024 and get all json files, 
    # the file is renamed in iclr2024.01022024.json, where 01022024 is in format MM/DD/YYYY
    # for a specific id, loop through all the json files and get the content of the json object
    # append the time code to the end of the json object and save the temporal to a new file
    
    root_folder = '/home/jyang/projects/papercopilot/logs/openreview/venues/iclr/iclr2024'
    target_id = 'HE9eUQlAvo'
    output_file = f'{root_folder}_{target_id}_temporal.json'

    full_pipeline(root_folder, tracing_threshold=1)
    # test(root_folder, target_id)