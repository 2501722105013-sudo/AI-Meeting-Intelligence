"""
Meeting Productivity Score Calculator.
Computes a dynamic 0-100 score based on decision count, task assignments,
speaker participation balance, and conversation metrics.
"""

import math
from typing import List, Dict, Any

def calculate_productivity_score(decisions: List[Dict[str, Any]], 
                                 tasks: List[Dict[str, Any]], 
                                 speaker_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate meeting productivity score (0-100) and provide metrics breakdown.
    """
    n_decisions = len(decisions)
    n_tasks = len(tasks)
    n_turns = len(speaker_turns)

    # 1. Decision & Action Item Score (Max 40 pts)
    # Target: At least 2 decisions and 2 tasks per meeting
    decision_score = min(20.0, n_decisions * 10.0)
    task_score = min(20.0, n_tasks * 8.0)
    action_dec_score = decision_score + task_score

    # 2. Task Assignment Completeness (Max 30 pts)
    # Tasks with named assignees and explicit deadlines
    complete_tasks = 0
    for t in tasks:
        person = t.get("assigned_person", "Unassigned")
        deadline = t.get("deadline", "TBD")
        if person != "Unassigned" and deadline not in ["TBD", "Next Sync"]:
            complete_tasks += 1
    
    clarity_ratio = (complete_tasks / n_tasks) if n_tasks > 0 else 0.5
    clarity_score = clarity_ratio * 30.0

    # 3. Speaker Participation Entropy / Balance (Max 30 pts)
    speaker_counts = {}
    for turn in speaker_turns:
        spk = turn.get("speaker", "Unknown")
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    total_turns = sum(speaker_counts.values())
    if total_turns > 0 and len(speaker_counts) > 1:
        # Calculate normalized Shannon entropy
        entropy = 0.0
        for spk, cnt in speaker_counts.items():
            p = cnt / total_turns
            entropy -= p * math.log2(p)
        max_entropy = math.log2(len(speaker_counts))
        participation_balance = entropy / max_entropy if max_entropy > 0 else 0.5
    else:
        participation_balance = 0.5

    participation_score = participation_balance * 30.0

    # Total Score (Bounded 0 to 100)
    total_score = round(min(100.0, action_dec_score + clarity_score + participation_score), 1)

    # Grade & Recommendation
    if total_score >= 85:
        grade = "A+ (Highly Productive)"
        feedback = "Exceptional meeting! Clear decisions made, action items assigned with deadlines, and balanced participation."
    elif total_score >= 70:
        grade = "B (Good & Effective)"
        feedback = "Productive session with good task ownership. Consider reserving time for key consensus building."
    elif total_score >= 50:
        grade = "C (Moderate)"
        feedback = "Moderate productivity. Recommend assigning explicit owners and deadlines for all tasks."
    else:
        grade = "D (Needs Improvement)"
        feedback = "Low decision density or single speaker dominance. Focus on clear action assignments."

    return {
        "overall_score": total_score,
        "grade": grade,
        "feedback": feedback,
        "breakdown": {
            "decisions_score": round(decision_score, 1),
            "tasks_score": round(task_score, 1),
            "clarity_score": round(clarity_score, 1),
            "participation_score": round(participation_score, 1),
            "speaker_participation_balance_pct": round(participation_balance * 100, 1)
        }
    }
