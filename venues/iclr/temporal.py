import os
import json
from collections import defaultdict
from typing import List, Dict
from scipy.optimize import linear_sum_assignment
import numpy as np

# v1
# def get_reviewer_footprints(data):
#     footprints = defaultdict(list)

#     for day in data:
#         ratings = list(map(int, day["rating"].split(';')))
#         confidences = list(map(int, day["confidence"].split(';')))
#         correctness = list(map(int, day["correctness"].split(';')))
#         novelty = list(map(int, day["technical_novelty"].split(';')))
#         time_code = day["time_code"]

#         for i in range(len(ratings)):
#             footprints[i].append({
#                 "time_code": time_code,
#                 "rating": ratings[i],
#                 "confidence": confidences[i],
#                 "correctness": correctness[i],
#                 "technical_novelty": novelty[i]
#             })

#     return dict(footprints)

# v2
# def parse_scores(scores: str) -> List[int]:
#     return list(map(int, scores.split(';')))

# def get_reviewer_profiles(day):
#     return [
#         (r, c, corr, n)
#         for r, c, corr, n in zip(
#             parse_scores(day['rating']),
#             parse_scores(day['confidence']),
#             parse_scores(day['correctness']),
#             parse_scores(day['technical_novelty'])
#         )
#     ]

# def trace_reviewers(data: List[Dict]) -> List[Dict]:
#     latest_day = data[-1]
#     canonical_profiles = get_reviewer_profiles(latest_day)
#     num_reviewers = len(canonical_profiles)

#     # Store mapping: day → [canonical_idx0, canonical_idx1, ...]
#     day_mappings = []

#     for day in data:
#         profiles = get_reviewer_profiles(day)
#         assigned = set()
#         day_map = [-1] * num_reviewers

#         for i, p in enumerate(profiles):
#             # Find the best matching canonical profile
#             for j, cp in enumerate(canonical_profiles):
#                 if j in assigned:
#                     continue
#                 if p == cp:  # exact match
#                     day_map[i] = j
#                     assigned.add(j)
#                     break
#         day_mappings.append({
#             "time_code": day["time_code"],
#             "reviewer_map": day_map  # maps current reviewer index → canonical reviewer ID
#         })

#     return day_mappings

# v3
# def parse_day(day):
#     return list(zip(
#         map(int, day["rating"].split(';')),
#         map(int, day["confidence"].split(';')),
#         map(int, day["correctness"].split(';')),
#         map(int, day["technical_novelty"].split(';'))
#     ))

# def trace_backwards(data):
#     n_days = len(data)
#     n_reviewers = len(data[-1]["rating"].split(';'))

#     # Canonical mapping on the last day (identity)
#     day_to_canonical_map = [None] * n_days
#     day_to_canonical_map[-1] = list(range(n_reviewers))

#     for day_idx in range(n_days - 2, -1, -1):
#         current_day = parse_day(data[day_idx])
#         next_day = parse_day(data[day_idx + 1])
#         next_map = day_to_canonical_map[day_idx + 1]

#         matched = [-1] * n_reviewers
#         used = set()

#         for i, curr in enumerate(current_day):
#             curr_conf = curr[1:]  # skip rating

#             best_match = -1
#             for j, nxt in enumerate(next_day):
#                 if j in used:
#                     continue
#                 if curr_conf == nxt[1:]:
#                     best_match = j
#                     break

#             if best_match != -1:
#                 matched[i] = next_map[best_match]
#                 used.add(best_match)

#         day_to_canonical_map[day_idx] = matched

#     # Pair with time_code
#     result = [{data[i]['time_code']: day_to_canonical_map[i]} for i in range(n_days)]
#     return result

# v4
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


if __name__ == "__main__":
    
    # loop through the root folder of iclr2024 and get all json files, 
    # the file is renamed in iclr2024.01022024.json, where 01022024 is in format MM/DD/YYYY
    # for a specific id, loop through all the json files and get the content of the json object
    # append the time code to the end of the json object and save the temporal to a new file
    
    root_folder = '/home/jyang/projects/papercopilot/logs/openreview/venues/iclr/iclr2024'
    target_id = 'HE9eUQlAvo'
    output_file = f'{root_folder}_{target_id}_temporal.json'

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
    
    # load json
    # with open(output_file.replace('.json', '_test.json'), 'r') as f:
        # results = json.load(f)
    
    # footprints = get_reviewer_footprints(results)
    footprints = trace_with_hungarian(results)
    
    # save footprints to a new file
    footprints_file = f'{root_folder}_{target_id}_footprints.json'