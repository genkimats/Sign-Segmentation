import json
import os

QUEUE_FILE = "train_queue.json"

def dequeue_jobs():
    if not os.path.exists(QUEUE_FILE):
        print("📭 Queue file does not exist. Nothing to dequeue.")
        return

    with open(QUEUE_FILE, "r") as f:
        try:
            queue = json.load(f)
        except json.JSONDecodeError:
            print("❌ Error reading queue file. It might be corrupted.")
            return

    # If the queue only contains the prefixes array (len <= 1), there are no jobs
    if not queue or len(queue) <= 1:
        print("📭 Queue is currently empty (no jobs to dequeue).")
        return

    prefix_tracker = queue[0].get("prefixes", [])
    jobs = queue[1:]

    print(f"\n{'='*75}")
    print("📋 CURRENT TRAINING QUEUE")
    print(f"{'='*75}")
    
    # Display the queue with temporary IDs
    for i, job in enumerate(jobs):
        prefix = job.get("prefix", "??")
        basename = job.get("basename", "Unknown Model")
        desc = job.get("description", "No description provided")
        
        print(f"[ID: {i}] Prefix {prefix} | {basename}")
        print(f"        └─ 📝 {desc}\n")

    print(f"{'='*75}")
    user_input = input("🗑️ Enter IDs to remove (comma-separated, e.g., 0, 2) or 'q' to quit: ").strip()

    if user_input.lower() == 'q' or not user_input:
        print("Canceled. Queue remains unchanged.")
        return

    try:
        # Parse inputs into a list of integers
        ids_to_remove = [int(x.strip()) for x in user_input.split(',')]
    except ValueError:
        print("❌ Invalid input. Please enter numbers separated by commas (e.g., 0, 1, 3).")
        return

    # Filter out valid IDs that actually exist in the list
    valid_ids = set([i for i in ids_to_remove if 0 <= i < len(jobs)])
    if not valid_ids:
        print("⚠️ No valid IDs matched. Queue remains unchanged.")
        return

    # Identify the prefixes of the removed jobs so we can free them up
    prefixes_to_remove = []
    for i in valid_ids:
        job_prefix = jobs[i].get("prefix")
        if job_prefix:
            try:
                prefixes_to_remove.append(int(job_prefix))
            except ValueError:
                pass

    # Clean up the tracker array at queue[0]
    new_prefixes = [p for p in prefix_tracker if p not in prefixes_to_remove]
    queue[0]["prefixes"] = new_prefixes

    # Rebuild the job queue by keeping only the non-selected IDs
    kept_jobs = [job for i, job in enumerate(jobs) if i not in valid_ids]
    
    # Reconstruct the final JSON payload
    new_queue = [queue[0]] + kept_jobs
    
    with open(QUEUE_FILE, "w") as f:
        json.dump(new_queue, f, indent=4)

    print(f"\n✅ Successfully removed {len(valid_ids)} jobs from the queue.")
    if prefixes_to_remove:
        print(f"♻️ Freed up prefixes: {prefixes_to_remove}")

if __name__ == "__main__":
    dequeue_jobs()