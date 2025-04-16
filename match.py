import os
import json
from datetime import datetime
from collections import defaultdict

FIELDS = ["confidence", "correctness", "technical_novelty"]
        
def get_date_from_filename(filename):
    try:
        date_str = filename.split(".")[1].replace(".json", "")
        return datetime.strptime(date_str, "%m%d%Y")
    except Exception as e:
        print(f"Filename format error: {filename}")
        raise e

def sort_files_by_date(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    files.sort(key=lambda x: get_date_from_filename(x))
    return files

def parse_scores(data_entry, fields=FIELDS):
    n_reviewers = len(data_entry[fields[0]].split(";"))
    reviewers = []
    for i in range(n_reviewers):
        scores = {}
        for field in fields:
            raw = data_entry.get(field, "")
            parts = raw.split(";") if raw else []
            score = int(parts[i]) if i < len(parts) and parts[i].isdigit() else None
            scores[field] = score
        reviewers.append(scores)
    return reviewers

def find_consistent_reviewers(folder, fields=FIELDS):
    sorted_files = sort_files_by_date(folder)

    first_file = sorted_files[0]
    with open(os.path.join(folder, first_file)) as f:
        data = json.load(f)
    first_day_data = {entry["id"]: entry for entry in data}
    # first_day_scores = {entry["id"]: parse_scores(entry) for entry in data}
    last_file = sorted_files[-1]
    with open(os.path.join(folder, last_file)) as f:
        data = json.load(f)
    last_day_data = {entry["id"]: entry for entry in data}
    # last_day_scores = {entry["id"]: parse_scores(entry) for entry in data}
    
    paper_matches = {}
    count = 0
    
    for paper_id, paper_scores in first_day_data.items():
        if paper_id not in last_day_data.keys():
            continue

        first_scores = parse_scores(paper_scores, fields)
        review_dict = last_day_data[paper_id]
        last_scores = parse_scores(review_dict, fields)

        data_set = set(json.dumps(d) for d in first_scores)
        first_scores = [json.loads(s) for s in data_set]
        data_set = set(json.dumps(d) for d in last_scores)
        last_scores = [json.loads(s) for s in data_set]
        
        matched = {}
        num_matches = 0

        for i, last_score in enumerate(last_scores):
            for j, first_score in enumerate(first_scores):
                if all(last_score[field] == first_score[field] for field in fields):
                    assert i not in matched.keys()
                    matched[i] = j
                    count += 1
                    break

        paper_matches[paper_id] = matched
    print(f"Total matches: {count}")
    return paper_matches

# Example usage
folder = "openreview/venues/iclr/iclr2024"
consistent_reviewers = find_consistent_reviewers(folder)

# Printing result for inspection
from pprint import pprint
pprint(dict(consistent_reviewers))


# I also have a script here for matching the data on the final day with the data from openreview (so that we can get the textual reviews)
# Before you run anything here, could you make sure you have the ICLR data from openreview
# I posted a public ICLR2024 data file here, stored in the form of [paper1, paper2, ...]
# Where each paper is stored as a dictionary {reviewer1: review1, reviewer2: review2, ...}
# https://huggingface.co/datasets/QiyaoWei/Openreview/tree/main

# import json
# with open("openreview/venues/iclr/iclr2024/iclr2024.01182024.json") as f:
#     data = json.load(f)
# day_data = {entry["id"]: entry for entry in data}

# with open("input_data.json") as f:
#     data = json.load(f)
# def convert_review_list_to_paper_dict(data):
#     last_day_data = {}

#     for paper_reviews in data:
#         # Get paper ID from any review (e.g., the first one)
#         if len(paper_reviews) == 0:
#             continue
#         first_reviewer = next(iter(paper_reviews))
#         assert len(paper_reviews[first_reviewer]) == 1
#         paper_id = paper_reviews[first_reviewer][0]["forum"]
        
#         if not paper_id:
#             continue  # Skip if no paper ID found
        
#         last_day_data[paper_id] = paper_reviews  # reviewer_number: review dict

#     return last_day_data

# last_day_data = convert_review_list_to_paper_dict(data)

# from typing import Dict

# FIELDS = ["rating", "confidence", "presentation"]

# def parse_scores(data_entry: Dict, fields=FIELDS):
#     """
#     Parses semicolon-separated score strings into a list of dicts (one per reviewer).
#     """
#     scores_by_reviewer = []
#     num_reviewers = len(data_entry[fields[0]].split(";"))

#     for i in range(num_reviewers):
#         reviewer_scores = {}
#         for field in fields:
#             raw = data_entry.get(field, "")
#             parts = raw.split(";") if raw else []
#             score = int(parts[i]) if i < len(parts) and parts[i].isdigit() else None
#             reviewer_scores[field] = score
#         scores_by_reviewer.append(reviewer_scores)
#     return scores_by_reviewer

# def clean_openreview_score(val):
#     # Extract numeric part from "6: Accept" or just return int if already clean
#     if isinstance(val, str):
#         return int(val[0].strip())
#     return int(val) if val is not None else None

# def extract_reviewer_vector(review_dict: Dict, fields=FIELDS):
#     """
#     Extracts a numerical score vector from a review_dict.
#     """
#     return {field: clean_openreview_score(review_dict.get(field)["value"]) for field in fields}

# def match_reviewers(data: Dict, last_day_data: Dict, fields=FIELDS):
#     paper_matches = {}

#     for paper_id, paper_scores in data.items():
#         if paper_id not in last_day_data.keys():
#             continue

#         first_scores = parse_scores(paper_scores, fields)
#         review_dict = last_day_data[paper_id]

#         matched = {}
#         used_indices = set()

#         for reviewer_id, review in review_dict.items():
#             review_vec = extract_reviewer_vector(review[0]["content"], fields)

#             # Only exact match
#             for idx, first_score in enumerate(first_scores):
#                 if idx in used_indices:
#                     continue
#                 if all(review_vec[field] == first_score[field] for field in fields):
#                     matched[reviewer_id] = idx
#                     used_indices.add(idx)
#                     break

#         paper_matches[paper_id] = matched

#     return paper_matches

# matches = match_reviewers(day_data, last_day_data)
# print(matches)
