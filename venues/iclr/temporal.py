import os
import json
import sys
from collections import defaultdict
from typing import List, Dict
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import csv
import re

year = 2025
if year == 2024:
    FIELDS = ["rating", "confidence", "correctness", "technical_novelty"] # iclr 2024
elif year == 2025:
    FIELDS = ["rating", "confidence", "soundness", "contribution", "presentation"] # iclr 2025
else:
    raise ValueError(f"Year {year} is not supported. Please update the FIELDS variable accordingly.")

class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            try:
                stream.write(message)
                stream.flush()
            except Exception:
                pass  # Ignore writing errors, like closed file

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass  # Ignore flushing errors, like closed file

def parse_day(day):
    return list(zip(*(map(int, day[field].split(';')) for field in FIELDS)))

def signature_cost(sig1, sig2):
    return sum(abs(a - b) for a, b in zip(sig1, sig2))

def sort_key(entry):
    mmddyyyy = entry['time_code']
    return mmddyyyy[4:] + mmddyyyy[:4]

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
    n_reviewers = len(data[-1][FIELDS[0]].split(';'))

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


def analyze_stability(
    results: Dict[str, List[Dict]],
    mode: str = "first_last",
    debug_dir: str = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Analyze the stability of reviewer profiles over time across papers.

    Stability is defined as a reviewer's score in specific dimensions (excluding "rating")
    not changing across snapshots.

    Parameters:
    ----------
    results : Dict[str, List[Dict]]
        A dictionary mapping paper IDs to a list of review snapshots across time.

    mode : str, optional (default="first_last")
        - "first_last": Compare only the first and last snapshot.
        - "all_days": Require the value to stay unchanged across all consecutive pairs.

    debug_dir : str or None (default=None)
        If provided, saves trace footprints (canonical reviewer ID alignments)
        using Hungarian algorithm into CSV files for each paper.

    Returns:
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        A summary dictionary with paper-level and reviewer-level stability stats.
        Includes counts, percentages, skipped papers metadata, and failed records.
    """
    total_papers = 0
    total_reviewers = 0

    # Dimensions excluding the primary sorting key (e.g., "rating")
    non_primary_fields = FIELDS[1:]
    
    # Counters for unchanged scores per dimension
    unchanged = {
        field: {"papers": 0, "reviewers": 0} for field in non_primary_fields
    }
    unchanged["all_non_rating"] = {"papers": 0, "reviewers": 0}

    metadata = {
        "fully_empty_profiles": 0,
        "papers_with_extra_reviewers": 0
    }

    failed_records = {}
    field_index = {field: idx for idx, field in enumerate(FIELDS)}

    for paper_id, entries in results.items():
        entries.sort(key=sort_key)
        if len(entries) <= 1:
            continue  # Only one snapshot — skip

        initial_raw = entries[0]
        final_raw = entries[-1]

        # Skip if any dimension is completely empty
        if any(initial_raw[field].strip() == '' for field in FIELDS):
            metadata["fully_empty_profiles"] += 1
            continue

        # Skip if the number of reviewers changes over time
        reviewer_counts = [len(entry[FIELDS[0]].split(';')) for entry in entries]
        if any(c != reviewer_counts[0] for c in reviewer_counts):
            metadata["papers_with_extra_reviewers"] += 1
            continue

        reviewer_count = reviewer_counts[0]
        total_papers += 1
        total_reviewers += reviewer_count

        reviewer_flags = [[True] * len(non_primary_fields) for _ in range(reviewer_count)]
        paper_flags = {field: True for field in non_primary_fields}

        # Compare scores across time
        if mode == "first_last":
            initial = parse_day(initial_raw)
            final = parse_day(final_raw)
            for i in range(reviewer_count):
                for j, field in enumerate(non_primary_fields):
                    idx = field_index[field]
                    if initial[i][idx] != final[i][idx]:
                        reviewer_flags[i][j] = False
                        paper_flags[field] = False
        elif mode == "all_days":
            for i in range(1, len(entries)):
                prev = parse_day(entries[i - 1])
                curr = parse_day(entries[i])
                for j in range(reviewer_count):
                    for k, field in enumerate(non_primary_fields):
                        idx = field_index[field]
                        if prev[j][idx] != curr[j][idx]:
                            reviewer_flags[j][k] = False
                            paper_flags[field] = False
        else:
            raise ValueError("Invalid mode. Use 'first_last' or 'all_days'.")

        # Aggregate stable dimensions
        for i, flag in enumerate(reviewer_flags):
            for j, field in enumerate(non_primary_fields):
                if flag[j]:
                    unchanged[field]["reviewers"] += 1
            if all(flag):
                unchanged["all_non_rating"]["reviewers"] += 1

        for field in non_primary_fields:
            if paper_flags[field]:
                unchanged[field]["papers"] += 1
        if all(paper_flags.values()):
            unchanged["all_non_rating"]["papers"] += 1
        else:
            failed_records[paper_id] = entries

        # --- Optional footprint debug output ---
        if all(paper_flags) and debug_dir :
            try:
                footprints = trace_with_hungarian(entries, max_cost_threshold=0)
                all_traced = all(
                    -1 not in trace.get(time_code, [])
                    for trace in footprints
                    for time_code in trace
                )
                filename = os.path.join(os.path.dirname(debug_dir), f'stable_{mode}', f"{paper_id}_{'success' if all_traced else 'fail'}.csv")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["time_code"] + FIELDS + ["canonical_ids"])
                    for i, day in enumerate(entries):
                        time_code = day["time_code"]
                        trace_map = list(footprints[i].values())[0]
                        row = [time_code] + [day[field] for field in FIELDS] + [";".join(map(str, trace_map))]
                        writer.writerow(row)
            except Exception as e:
                print(f"⚠️ Error writing debug CSV for {paper_id}: {e}")

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
                } for key in list(non_primary_fields) + ["all_non_rating"]
            }
        },
        "reviewer_level": {
            "total_reviewers": total_reviewers,
            "unchanged": {
                key: {
                    "count": unchanged[key]["reviewers"],
                    "percentage": percentage(unchanged[key]["reviewers"], total_reviewers)
                } for key in list(non_primary_fields) + ["all_non_rating"]
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
    """
    Prints a side-by-side comparison table showing stability differences
    between "first_last" and "all_days" modes, at either the paper or reviewer level.

    Parameters:
    ----------
    first : Dict
        Output from analyze_stability in "first_last" mode.

    all_days : Dict
        Output from analyze_stability in "all_days" mode.

    level : str
        Either "paper_level" or "reviewer_level".

    Output:
    -------
    A stylized, color-coded comparison table printed to stdout.
    Each row represents a review dimension (e.g., confidence, correctness),
    showing:
        - # and % of stable entities (papers or reviewers) in first_last mode
        - # and % of stable entities in all_days mode
        - Percentage point difference between them
    """
    headers = ["Dimension", "First & Last", "All Days", "Δ Difference"]
    rows = []

    for key in first[level]["unchanged"]:
        # Format dimension name
        key_label = key.replace('_', ' ').capitalize()

        # Extract counts and percentages
        f_val = first[level]["unchanged"][key]
        a_val = all_days[level]["unchanged"][key]
        f_str = f"{f_val['count']} ({f_val['percentage']}%)"
        a_str = f"{a_val['count']} ({a_val['percentage']}%)"

        # Compute difference
        diff = round(f_val["percentage"] - a_val["percentage"], 2)
        diff_str = f"{diff:+.2f}%"
        diff_color = "1;32" if diff >= 0 else "1;31"

        # Compose table row with colors
        rows.append([
            print_colored(key_label, "1;36"),
            print_colored(f_str, "1;33"),
            print_colored(a_str, "1;35"),
            print_colored(diff_str, diff_color)
        ])

    # Print comparison block
    print(print_colored(f"\n📊 Comparison Table — {level.replace('_', ' ').capitalize()}", "1;34"))
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def print_block(stats):
    """
    Prints a detailed stability analysis block based on the result
    from `analyze_stability`.

    Includes metadata summary, paper- and reviewer-level stats,
    and a short explanation of the stability checking method.

    Parameters:
    ----------
    stats : Dict
        Dictionary returned from `analyze_stability`, containing:
            - mode ("first_last" or "all_days")
            - paper_level and reviewer_level stats
            - paper_counts and metadata
    """

    mode_str = "First & Last Snapshot" if stats['mode'] == "first_last" else "Across All Days"
    print(print_colored(f"\n=== Stability Analysis ({mode_str}) ===", "1;36"))

    # Show method explanation
    if stats['mode'] == "first_last":
        print(print_colored("🧠 This mode compares only the first and last snapshots of each paper's review timeline.\n"
                            "If a reviewer's score remained the same from beginning to end, it is considered stable — even if it changed temporarily in between.\n", "1;37"))
    else:
        print(print_colored("🔍 This mode checks reviewer stability across all snapshots in the timeline.\n"
                            "A reviewer's score must remain unchanged at all time points to be considered stable.\n", "1;37"))

    # Print paper count summary
    print(print_colored("Paper Counts Breakdown:", "1;33"))
    print(f"  {print_colored('Valid:', '1;33')} {stats['paper_counts']['valid']} papers — included in the stability evaluation")
    print(f"  {print_colored('Fully empty profiles:', '1;33')} {stats['paper_counts']['fully_empty_profiles']} papers — skipped due to missing scores")
    print(f"  {print_colored('Extra reviewers:', '1;33')} {stats['paper_counts']['extra_reviewers']} papers — skipped due to reviewer number inconsistencies")
    print(print_colored(f"\nTotal evaluated papers: {stats['paper_counts']['total_evaluated']}", "1;33"))
    print(print_colored(f"Total reviewers analyzed: {stats['reviewer_level']['total_reviewers']}", "1;33"))

    # Paper-level stability block
    print(print_colored("\n📘 Paper-Level Stability:", "1;34"))
    print(print_colored("Each paper is marked as stable in a dimension only if all reviewers remained stable in that dimension.", "0;37"))
    for key, values in stats['paper_level']['unchanged'].items():
        print(f"  {print_colored(key.capitalize(), '1;32')}: "
              f"{values['count']} papers ({values['percentage']}%)")

    # Reviewer-level stability block
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
                    writer.writerow(["time_code"] + FIELDS + ["canonical_ids"])
                    for i, day in enumerate(records):
                        time_code = day["time_code"]
                        trace_map = list(footprints[i].values())[0]
                        row = [time_code] + [day[field] for field in FIELDS] + [";".join(map(str, trace_map))]
                        writer.writerow(row)

        except Exception as e:
            print(f"⚠️ Error tracing paper {paper_id}: {e}")

    print(f"\n✅ Successfully traced {success} out of {total} failed papers ({round(success / total * 100, 2)}%) using max_cost_threshold = {max_cost_threshold}")
    return traced_entries

    
def full_pipeline(root_folder: str, tracing_threshold: int = 3, debug_folder=None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Executes the full pipeline for analyzing review stability across time,
    printing diagnostics, and optionally tracing failed review profiles using Hungarian matching.

    Steps:
        1. Load and group review entries by paper ID
        2. Run stability analysis ("first_last" and "all_days" modes)
        3. Print analysis blocks and comparison tables
        4. Run reviewer ID tracing on failed records
        5. Return result summaries

    Parameters:
    ----------
    root_folder : str
        Directory containing review snapshots in the form: iclrYYYY.MMDDYYYY.json

    tracing_threshold : int (default=3)
        Cost threshold for reviewer tracing using the Hungarian algorithm

    debug_folder : str (optional)
        If provided, stores footprint matching logs and debug CSVs

    Returns:
    -------
    Dict[str, Dict]
        {
            "first_last": {...},      # Result from first vs last snapshot comparison
            "all_days": {...},        # Result from all days comparison
            "traced_success": {...}   # Successfully traced failed papers
        }
    """

    id_to_entries = defaultdict(list)

    # Step 1: Load entries grouped by paper ID
    for filename in os.listdir(root_folder):
        if filename.endswith('.json') and filename.startswith(f'iclr{year}'):
            time_code = filename.split('.')[1]
            with open(os.path.join(root_folder, filename), 'r') as f:
                data = json.load(f)
                for entry in data:
                    if entry.get('id') and all(field in entry for field in FIELDS):
                        entry['time_code'] = time_code
                        id_to_entries[entry['id']].append(entry)

    # Filter papers that have more than one snapshot
    id_to_filtered_entries = {
        paper_id: sorted(entries, key=sort_key)
        for paper_id, entries in id_to_entries.items()
        if len(entries) > 1
    }

    # Step 2: Run stability analysis (first-last and full timeline)
    stats_first_last = analyze_stability(id_to_filtered_entries, mode="first_last")
    stats_all_days = analyze_stability(id_to_filtered_entries, mode="all_days", debug_dir=debug_folder)

    # Step 3: Print summaries
    print_block(stats_first_last)
    print_block(stats_all_days)
    print_comparison_table(stats_first_last, stats_all_days, level="paper_level")
    print_comparison_table(stats_first_last, stats_all_days, level="reviewer_level")

    # Step 4: Trace failed records using Hungarian algorithm
    failed_records = stats_all_days["failed_records"]
    traced_entries = trace_failed_records(
        failed_records,
        max_cost_threshold=tracing_threshold,
        debug_dir=debug_folder
    )

    # Step 5: Return summary results
    return {
        "first_last": stats_first_last,
        "all_days": stats_all_days,
        "traced_success": traced_entries,
        "total_evaluated": stats_all_days["paper_counts"]["total_evaluated"],
    }
    
def generate_tracing_summary(debug_root: str, thresholds: range = range(0, 6), output_file: str = "summary.csv"):
    """
    Generate a summary CSV indicating tracing success (O) or failure (X) for each paper across different thresholds.
    Also appends two rows beneath the header:
        - One with raw counts: 2041/3120
        - One with percentages: 65.45%

    Parameters:
    ----------
    debug_root : str
        Base path where traced CSVs are saved, expected subfolders: *_footprints_threshold_{i}

    thresholds : range
        Threshold values to check (default: 0 to 5 inclusive)

    output_file : str
        Path to save the summary CSV file
    """
    paper_ids = set()
    results_by_threshold = {}
    count_row = ["summary_count"]
    percentage_row = ["summary_percent"]

    for threshold in thresholds:
        folder = f"{debug_root}/threshold_{threshold}"
        if not os.path.exists(folder):
            count_row.append("")
            percentage_row.append("")
            continue

        results = {}
        for filename in os.listdir(folder):
            if filename.endswith(".csv") and ("_success" in filename or "_fail" in filename):
                paper_id, result = filename.replace(".csv", "").split("_")
                results[paper_id] = "O" if result == "success" else "X"
                paper_ids.add(paper_id)
        results_by_threshold[threshold] = results

        # Read tracing success summary from log
        log_file = f"{debug_root}/threshold_{threshold}_log.txt"
        if not os.path.exists(log_file):
            print(f"⚠️ Log file not found for threshold {threshold}: {log_file}")
            continue
        # Extract the summary line from the log file
        count = ""
        percent = ""
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f:
                    if "Successfully traced" in line:
                        match = re.search(r"(\d+) out of (\d+).*?\(([\d.]+)%\)", line)
                        if match:
                            traced, total, pct = match.groups()
                            count = f"{traced}/{total}"
                            percent = f"{pct}%"
                        break
        count_row.append(count)
        percentage_row.append(percent)

    # Prepare summary data
    sorted_papers = sorted(paper_ids)
    header = ["paper_id"] + [f"threshold_{t}" for t in thresholds]
    rows = []

    for paper_id in sorted_papers:
        row = [paper_id]
        for t in thresholds:
            row.append(results_by_threshold.get(t, {}).get(paper_id, ""))
        rows.append(row)

    # Write to CSV using csv.writer
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(count_row)
        writer.writerow(percentage_row)
        writer.writerows(rows)

    print(f"✅ Tracing summary saved to {output_file}")
    return output_file

def test(root_folder, target_id='HE9eUQlAvo'):
    
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
    # output_file = f'{root_folder}_{target_id}_temporal.json'
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
    
    root_folder = f'/home/jyang/projects/papercopilot/logs/openreview/venues/iclr/iclr{year}'
    
    tracing_threshold_min = 0
    tracing_threshold_max = 6
    tracing_threshold_save = 1
    print(f"Tracing thresholds: {tracing_threshold_min} to {tracing_threshold_max}")
    
    for t in range(tracing_threshold_min, tracing_threshold_max):
        debug_folder = root_folder + f"/footprints/threshold_{t}"
        os.makedirs(debug_folder, exist_ok=True)
        with open(debug_folder +'_log.txt', 'w') as logfile:
            sys.stdout = TeeOutput(sys.__stdout__, logfile)
            
            print(f"cat this {debug_folder}_log.txt in a terminal to see the output with colors")
            print(f"Debug folder: {debug_folder}")

            result = full_pipeline(root_folder, tracing_threshold=t, debug_folder=debug_folder)
            # test(root_folder, target_id)
            
    print("\n📊 Generating final summary table...")
    generate_tracing_summary(
        debug_root=root_folder + "/footprints",
        thresholds=range(0, 6),
        output_file=root_folder + "/footprints/tracing_summary.csv"
    )
    
    def footprint_csv2list(csv_path):
        with open(footprint_path, 'r') as f:
            footprint_data = list(csv.reader(f))
            # remove the header
            footprint_data = footprint_data
            # convert to dict
            tracing_footprint = []
            for row in footprint_data:
                tracing_footprint.append(row)
        return tracing_footprint
    
    # load the meta data and prepare concate the tracing result and save to the new json file
    meta_path = f"{root_folder}.json"
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
        
    # loop through records and mark the status for each record that pass the stability test
    # Stability is defined as a reviewer's score in specific dimensions (excluding "rating", the sorting dimension) not changing across snapshots.
    stability_path = f"{root_folder}/footprints/stable_all_days"
    stability_tests = os.listdir(stability_path)
    stability_tests = [f.split('_success')[0] for f in stability_tests if f.endswith('.csv') and 'success' in f]
    stability_tests_passed_id = set(stability_tests)
    for i, paper in enumerate(meta_data):
        if paper['id'] in stability_tests_passed_id:
            meta_data[i]['tracing_score'] = -1 # stable and success
            
            # load the footprint and include in the json
            footprint_path = os.path.join(stability_path, f"{paper['id']}_success.csv")
            tracing_footprint = footprint_csv2list(footprint_path)
            meta_data[i]['tracing_footprint'] = tracing_footprint
        else:
            meta_data[i]['tracing_score'] = np.inf # marked as unstable
        
    # load the tracing result for unstable records
    tracing_path = f"{root_folder}/footprints"
    with open(os.path.join(tracing_path, 'tracing_summary.csv'), 'r') as f:
        tracing_data = list(csv.reader(f))
        
    # load the tracing result and find and update the threshold that can pass the tracing
    tracing_results = {}
    for row in tracing_data[3:]:
        paper_id = row[0]
        for i in range(1, len(row)):
            if row[i] == 'O':
                tracing_results[paper_id] = i - 1
                break
    for i, paper in enumerate(meta_data):
        if paper['id'] in tracing_results:
            theshold = tracing_results[paper['id']]
            meta_data[i]['tracing_score'] = theshold
            
            # load the footprint and include in the json
            footprint_path = os.path.join(tracing_path, f"threshold_{theshold}", f"{paper['id']}_success.csv")
            tracing_footprint = footprint_csv2list(footprint_path)
            meta_data[i]['tracing_footprint'] = tracing_footprint
        else:
            # no need to modify the score
            pass
            
    # check if all records in meta_data have tracing_score
    all_records = True
    for paper in meta_data:
        if 'tracing_score' not in paper:
            all_records = False
            break
    if not all_records:
        print("Not all records have tracing_score, please check the tracing result.")
    else:
        # print the tracing_score count
        print("All records have tracing_score.")
        tracing_score_count = {}
        for paper in meta_data:
            if paper['tracing_score'] not in tracing_score_count:
                tracing_score_count[paper['tracing_score']] = 0
            tracing_score_count[paper['tracing_score']] += 1
        print("Tracing score count:")
        for key in sorted(tracing_score_count.keys()):
            print(f"  {key}: {tracing_score_count[key]}")
            
    # loop through the meta data and save the record that have the status pass save_threshold to the new file
    meta2save = []
    for paper in meta_data:
        if paper['tracing_score'] <= tracing_threshold_save:
            meta2save.append(paper)
    # percentage = len(meta2save) / result['total_evaluated'] * 100
    percentage = len(meta2save) / len(meta_data) * 100
    output_file = f"{root_folder}_threshold{tracing_threshold_save}_{len(meta2save)}_records.json"
    # with open(output_file, 'w') as f:
        # json.dump(meta2save, f, indent=4)
    # print(f"Saved {len(meta2save)} records to {output_file}, with tracing_score <= {tracing_threshold_save}, {percentage:.2f}% in total {result['total_evaluated']} evaluated records")
    print("Done!")
    
    # cleanup
    meta2save_reviewers = []
    for paper in tqdm(meta_data):
        
        paper_new = {
            'id': paper['id'],
            'title': paper['title'],
            'tracing_score': paper['tracing_score'],
            'review': {}
        }
        
        reviewers = paper['reviewers'].split(';')
        
        if 'tracing_footprint' not in paper:
            # tracing failed
            meta2save_reviewers.append(paper_new)
            continue
        
        review_dims = paper['tracing_footprint'][0]
        assert review_dims[1:-1] == FIELDS
        
        # showcase only first two dimension in the fields
        review_dims = FIELDS[:2]
        
        # loop through all reviewers and get the review profile
        for i, reviewer in enumerate(reviewers):
            paper_new['review'][reviewer] = {}
            for r, review_dim in enumerate(review_dims):
                paper_new['review'][reviewer][review_dim] = []
                for t, footprint in enumerate(paper['tracing_footprint']):
                    canonical_id = footprint[-1].split(';')
                    if t == 0: continue # skip the first row, which is the header
                    if paper['tracing_score'] <= tracing_threshold_save:
                        paper_new['review'][reviewer][review_dim].append(footprint[r+1].split(';')[canonical_id.index(str(i))])
                    else:
                        paper_new['review'][reviewer][review_dim].append(-1)
                
                # output only the first and last
                first_last_only = True
                if first_last_only:
                    first, last = paper_new['review'][reviewer][review_dim][0], paper_new['review'][reviewer][review_dim][-1]
                    paper_new['review'][reviewer][review_dim] = f"{first};{last}"
                else:
                    paper_new['review'][reviewer][review_dim] = ';'.join(paper_new['review'][reviewer][review_dim])
        
        meta2save_reviewers.append(paper_new)
        
    # save the meta2save_reviewers to a new file
    output_file = f"{root_folder}_threshold{tracing_threshold_save}_{len(meta2save_reviewers)}_reviewers.json"
    with open(output_file, 'w') as f:
        json.dump(meta2save_reviewers, f, indent=4)
    print("Done!")
    