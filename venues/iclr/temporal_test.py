from scipy.optimize import linear_sum_assignment
import numpy as np

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

example = [
    {
        "id": "HE9eUQlAvo",
        "rating": "3;5;6;6;6",
        "confidence": "4;4;3;4;4",
        "correctness": "2;3;3;3;3",
        "technical_novelty": "2;3;3;3;3",
        "time_code": "11182023"
    },
    {
        "id": "HE9eUQlAvo",
        "rating": "5;5;6;6;6",
        "confidence": "4;4;3;4;4",
        "correctness": "3;3;3;3;3",
        "technical_novelty": "3;3;3;3;3",
        "time_code": "11192023"
    },
    {
        "id": "HE9eUQlAvo",
        "rating": "5;6;6;6;8",
        "confidence": "4;4;4;4;3",
        "correctness": "3;3;3;3;3",
        "technical_novelty": "3;3;3;3;3",
        "time_code": "11202023"
    }
]

result = trace_with_hungarian(example)
import json
print(json.dumps(result, indent=2))
