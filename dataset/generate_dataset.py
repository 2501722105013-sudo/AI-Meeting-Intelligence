"""
Synthetic Dataset Generator for AI Meeting Intelligence & Decision Tracker.
Generates 2,000 realistic enterprise meeting transcripts across 6 corporate domains.
Splits data into 70% Train, 15% Validation, 15% Test.
"""

import json
import random
import os
import csv
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)

DOMAINS = [
    "Software Development",
    "Project Planning",
    "Marketing",
    "HR",
    "Finance",
    "Product Management"
]

PARTICIPANTS_POOL = {
    "Software Development": ["Alex Chen (Lead)", "Sarah Jenkins (Senior Dev)", "David Miller (DevOps)", "Priya Sharma (QA)", "Marcus Vance (Backend Dev)", "Elena Rostova (Frontend Dev)"],
    "Project Planning": ["Rachel Green (Project Manager)", "Tom Hanks (Scrum Master)", "Carlos Mendez (Operations)", "Jessica Alba (Resource Mgr)", "Brian Cox (Risk Analyst)"],
    "Marketing": ["Emily Watson (CMO)", "James Wilson (Content Lead)", "Sophia Martinez (SEO Specialist)", "Daniel Craig (Campaign Lead)", "Olivia Wilde (Social Media)"],
    "HR": ["Karen Smith (HR Director)", "Michael Scott (Talent Acquisition)", "Pam Beesly (Employee Relations)", "Jim Halpert (HR Business Partner)", "Dwight Schrute (Compliance)"],
    "Finance": ["Robert Sterling (CFO)", "Laura Croft (Financial Analyst)", "Bruce Wayne (Auditor)", "Diana Prince (Budget Controller)", "Clark Kent (Accountant)"],
    "Product Management": ["Samantha Reed (VP Product)", "Kevin Bacon (Product Manager)", "Uma Thurman (UX Researcher)", "Ethan Hunt (Product Analyst)", "Grace Hopper (Tech PM)"]
}

TOPICS_POOL = {
    "Software Development": ["API Migration", "Microservice Architecture", "CI/CD Pipeline Optimization", "Bug Triage", "Code Review Standards", "Database Indexing", "Cloud Infrastructure"],
    "Project Planning": ["Q3 Milestone Alignment", "Sprint Planning", "Resource Allocation", "Risk Mitigation Strategy", "Timeline Estimation", "Vendor Onboarding"],
    "Marketing": ["Q4 Ad Campaign", "Brand Refresh Strategy", "Customer Acquisition Cost", "Product Launch Event", "Social Media Engagement", "Content Marketing Roadmap"],
    "HR": ["Quarterly Performance Reviews", "Hybrid Work Policy", "Employee Onboarding Redesign", "Compensation Benchmark", "Diversity and Inclusion Seminar"],
    "Finance": ["Annual Budget Allocation", "Cost Reduction Program", "Revenue Growth Projection", "Audit Compliance", "Software Subscription Review"],
    "Product Management": ["User Feedback Analysis", "Feature Prioritization Matrix", "Mobile App Redesign", "Competitor Benchmarking", "Churn Rate Reduction"]
}

DECISION_TEMPLATES = [
    "We agreed to finalize the {topic} proposal by {date}.",
    "It was decided that {person} will lead the implementation of {topic}.",
    "The team confirmed the approval for the {topic} budget.",
    "We officially decided to postpone {topic} until next quarter.",
    "It was finalized that we will migrate our systems to {topic}.",
    "The board approved the plan presented by {person} regarding {topic}.",
    "We agreed upon prioritizing {topic} over secondary features.",
    "The executive committee confirmed the timeline for {topic}."
]

TASK_TEMPLATES = [
    {"task": "Draft technical specification for {topic}", "priority": "High"},
    {"task": "Prepare financial forecast report for {topic}", "priority": "High"},
    {"task": "Schedule follow-up sync with stakeholders on {topic}", "priority": "Medium"},
    {"task": "Conduct user research survey regarding {topic}", "priority": "Medium"},
    {"task": "Review security compliance checklist for {topic}", "priority": "High"},
    {"task": "Set up monitoring dashboard for {topic}", "priority": "Low"},
    {"task": "Create training presentation for team on {topic}", "priority": "Medium"},
    {"task": "Update documentation and sprint backlog for {topic}", "priority": "Low"}
]

INTERRUPTIONS = [
    " -- wait, sorry to interrupt, but ",
    " -- hold on a second, ",
    " -- if I can just jump in real quick, ",
    " -- pardon me for breaking in, but ",
    " -- before we move past that point, "
]

AGREEMENTS = [
    "I completely agree with that direction.",
    "That makes total sense from an operational standpoint.",
    "Sounds solid to me, let's proceed.",
    "I second that proposal.",
    "Good point. I'm aligned with this approach."
]

DISAGREEMENTS = [
    "I have some concerns about the timeline.",
    "Are we sure we have enough budget for this?",
    "I don't think that's realistic given our current velocity.",
    "Let's be cautious before committing to that schedule.",
    "I'm not fully convinced that solves the core problem."
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "next week", "end of month", "by end of quarter"]

def clean_speaker(full_name):
    return full_name.split()[0]

def generate_meeting(meeting_index):
    dept = random.choice(DOMAINS)
    meeting_id = f"MTG_{dept[:4].upper()}_{meeting_index:04d}"
    
    start_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 300))
    date_str = start_date.strftime("%Y-%m-%d")
    
    participants_full = random.sample(PARTICIPANTS_POOL[dept], k=random.randint(3, min(5, len(PARTICIPANTS_POOL[dept]))))
    speakers = [clean_speaker(p) for p in participants_full]
    
    selected_topics = random.sample(TOPICS_POOL[dept], k=random.randint(2, 4))
    
    num_sentences = random.randint(15, 40)
    transcript_lines = []
    
    # Track generated metadata
    decisions = []
    tasks = []
    deadlines = []
    
    # Opening
    lead_speaker = speakers[0]
    transcript_lines.append(f"{lead_speaker}: Welcome everyone to today's {dept} sync regarding {selected_topics[0]}.")
    transcript_lines.append(f"{speakers[1]}: Thanks {lead_speaker}. Let's make sure we cover {', '.join(selected_topics)} today.")
    
    # Conversation generation
    current_topic = selected_topics[0]
    sentence_count = 2
    
    while sentence_count < num_sentences - 3:
        spk = random.choice(speakers)
        
        # Decide turn type
        turn_type = random.choices(["discussion", "agreement", "disagreement", "decision", "task"], weights=[0.4, 0.2, 0.15, 0.12, 0.13])[0]
        
        if turn_type == "discussion":
            inter = random.choice(INTERRUPTIONS) if random.random() < 0.25 else " "
            line = f"{spk}:{inter}We need to carefully evaluate {current_topic} to ensure quality deliverables."
        elif turn_type == "agreement":
            line = f"{spk}: {random.choice(AGREEMENTS)}"
        elif turn_type == "disagreement":
            line = f"{spk}: {random.choice(DISAGREEMENTS)}"
        elif turn_type == "decision":
            day_deadline = random.choice(DAYS)
            decision_text = random.choice(DECISION_TEMPLATES).format(topic=current_topic, date=day_deadline, person=spk)
            line = f"{spk}: {decision_text}"
            decisions.append({
                "decision": f"{current_topic} - {decision_text}",
                "date": day_deadline,
                "person": spk
            })
            deadlines.append(day_deadline)
        elif turn_type == "task":
            assigned_spk = random.choice(speakers)
            deadline_val = random.choice(DAYS)
            task_tmpl = random.choice(TASK_TEMPLATES)
            task_desc = task_tmpl["task"].format(topic=current_topic)
            priority_val = task_tmpl["priority"]
            
            line = f"{spk}: {assigned_spk} will {task_desc.lower()} by {deadline_val}."
            tasks.append({
                "task": task_desc,
                "assigned_person": assigned_spk,
                "deadline": deadline_val,
                "priority": priority_val,
                "status": "Pending"
            })
            deadlines.append(deadline_val)
            
        transcript_lines.append(line)
        sentence_count += 1
        
        # Switch topic occasionally
        if random.random() < 0.2 and len(selected_topics) > 1:
            current_topic = random.choice(selected_topics)
            
    # Guarantee at least 1 decision and 1 task per meeting if empty
    if not decisions:
        spk = speakers[0]
        day_deadline = random.choice(DAYS)
        decision_text = f"We decided to finalize the strategy for {selected_topics[0]} by {day_deadline}."
        transcript_lines.append(f"{spk}: {decision_text}")
        decisions.append({
            "decision": f"{selected_topics[0]} strategy approval",
            "date": day_deadline,
            "person": spk
        })
        deadlines.append(day_deadline)

    if not tasks:
        spk = speakers[1] if len(speakers) > 1 else speakers[0]
        day_deadline = random.choice(DAYS)
        task_desc = f"Prepare action plan for {selected_topics[0]}"
        transcript_lines.append(f"{speakers[0]}: {spk} will {task_desc.lower()} by {day_deadline}.")
        tasks.append({
            "task": task_desc,
            "assigned_person": spk,
            "deadline": day_deadline,
            "priority": "High",
            "status": "Pending"
        })
        deadlines.append(day_deadline)

    # Closing
    transcript_lines.append(f"{lead_speaker}: Great discussion today. Let's execute on these action items and sync next week.")
    
    full_transcript = "\n".join(transcript_lines)
    
    # Unique deadlines deduplicated
    unique_deadlines = list(set(deadlines))
    
    return {
        "meeting_id": meeting_id,
        "date": date_str,
        "department": dept,
        "participants": participants_full,
        "speaker_names": speakers,
        "timestamp": f"{random.randint(20, 60)} minutes",
        "transcript": full_transcript,
        "tasks_assigned": tasks,
        "decisions_made": decisions,
        "deadlines": unique_deadlines,
        "topics_discussed": selected_topics
    }

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating 2,000 synthetic enterprise meeting transcripts...")
    all_meetings = [generate_meeting(i + 1) for i in range(2000)]
    
    # Shuffle for random distribution across splits
    random.shuffle(all_meetings)
    
    n_total = len(all_meetings)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_data = all_meetings[:n_train]
    val_data = all_meetings[n_train:n_train + n_val]
    test_data = all_meetings[n_train + n_val:]
    
    print(f"Dataset split sizes -> Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Save JSON files
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
        
    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
        
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
        
    # Generate tabular CSV dataset representation for EDA
    csv_file = os.path.join(output_dir, "full_dataset.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "meeting_id", "date", "department", "duration_minutes", 
            "speaker_count", "sentence_count", "decision_count", 
            "task_count", "primary_topic", "full_transcript"
        ])
        
        for m in all_meetings:
            dur = int(m["timestamp"].split()[0])
            speakers = len(m["speaker_names"])
            sentences = len(m["transcript"].split("\n"))
            decisions = len(m["decisions_made"])
            tasks = len(m["tasks_assigned"])
            prim_topic = m["topics_discussed"][0] if m["topics_discussed"] else "General"
            
            writer.writerow([
                m["meeting_id"], m["date"], m["department"], dur,
                speakers, sentences, decisions, tasks, prim_topic, m["transcript"]
            ])
            
    print(f"Dataset generation complete! Saved files to {output_dir}")
    
    # Generate sample transcript text files for test upload
    sample_dir = os.path.join(output_dir, "sample_transcripts")
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_txt = os.path.join(sample_dir, "sample_software_meeting.txt")
    with open(sample_txt, "w", encoding="utf-8") as f:
        f.write(all_meetings[0]["transcript"])
        
    print(f"Sample transcript written to {sample_txt}")

if __name__ == "__main__":
    main()
