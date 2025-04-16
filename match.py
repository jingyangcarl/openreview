import os
import json
from datetime import datetime
from collections import defaultdict

FIELDS = ["rating", "confidence", "presentation"]

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

def find_consistent_reviewers(folder):
    sorted_files = sort_files_by_date(folder)
    consistent_reviewers = defaultdict(lambda: defaultdict(dict))  # paper_id -> reviewer_id -> reviewer_score

    all_reviewers_status = defaultdict(lambda: defaultdict(list))  # paper_id -> reviewer_id -> [scores across files]

    for file in sorted_files:
        with open(os.path.join(folder, file), "r") as f:
            papers = json.load(f)

        for paper in papers:
            paper_id = paper["id"]
            reviewers = parse_scores(paper)

            # Collect reviewers' scores across all files for later comparison
            for reviewer_id, reviewer in enumerate(reviewers):
                all_reviewers_status[paper_id][reviewer_id].append(reviewer)

    # Now filter reviewers that are consistent across all files
    for paper_id, reviewers in all_reviewers_status.items():
        for reviewer_id, scores in reviewers.items():
            # Check if the reviewer has the same confidence and presentation across all files
            confidence_set = {score["confidence"] for score in scores}
            presentation_set = {score["presentation"] for score in scores}

            # If confidence and presentation are the same across all files, mark as consistent
            if len(confidence_set) == 1 and len(presentation_set) == 1:
                consistent_reviewers[paper_id][reviewer_id] = scores[0]  # Save the reviewer's score

    return consistent_reviewers

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

# with open("iclr2024.json") as f:
#     data = json.load(f)
# def convert_review_list_to_paper_dict(data):
#     final_data = {}

#     for paper_reviews in data:
#         # Get paper ID from any review (e.g., the first one)
#         if len(paper_reviews) == 0:
#             continue
#         first_reviewer = next(iter(paper_reviews))
#         assert len(paper_reviews[first_reviewer]) == 1
#         paper_id = paper_reviews[first_reviewer][0]["forum"]
        
#         if not paper_id:
#             continue  # Skip if no paper ID found
        
#         final_data[paper_id] = paper_reviews  # reviewer_number: review dict

#     return final_data

# final_data = convert_review_list_to_paper_dict(data)

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

# def match_reviewers(data: Dict, final_data: Dict, fields=FIELDS):
#     paper_matches = {}

#     for paper_id, paper_scores in data.items():
#         if paper_id not in final_data.keys():
#             continue

#         score_vectors = parse_scores(paper_scores, fields)
#         review_dict = final_data[paper_id]

#         matched = {}
#         used_indices = set()

#         for reviewer_id, review in review_dict.items():
#             review_vec = extract_reviewer_vector(review[0]["content"], fields)

#             # Only exact match
#             for idx, score_vec in enumerate(score_vectors):
#                 if idx in used_indices:
#                     continue
#                 if all(review_vec[field] == score_vec[field] for field in fields):
#                     matched[reviewer_id] = idx
#                     used_indices.add(idx)
#                     break
#                 matched[reviewer_id] = best_idx
#                 used_indices.add(best_idx)

#         paper_matches[paper_id] = matched

#     return paper_matches

# matches = match_reviewers(day_data, final_data)
# print(matches)
