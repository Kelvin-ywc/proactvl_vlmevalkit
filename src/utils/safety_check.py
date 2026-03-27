import os
import json
import multiprocessing as mp
from typing import Tuple, Dict, Any
import re
from pathlib import Path
from collections import Counter, defaultdict

from azure.identity import get_bearer_token_provider, AzureCliCredential
import openai

# ========================
# Config
# ========================
ENDPOINT = "https://haotian-east-us-2.openai.azure.com/"
SCOPE = "https://cognitiveservices.azure.com/.default"
API_VERSION = "2024-12-01-preview"
MODEL = "gpt-4o-3"

FILE_PATH = "/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/DATA/ann/all_in_one_val.jsonl"

STATE_DIR = "/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/src/utils/_safety_state"
PROGRESS_PATH = os.path.join(STATE_DIR, "progress.json")
BLOCKED_PATH = os.path.join(STATE_DIR, "blocked.jsonl")

PROMPT_TEMPLATE_v1 = """You are a strict-but-lenient word-level compliance checker for an open-source gaming dataset.
Your ONLY job is to detect whether the JSON sample contains clearly disallowed WORDS/PHRASES in these categories:
1) PROFANITY/VULGARITY: explicit swear words or crude vulgar terms (including obfuscated forms like f**k, s#it, f.u.c.k, leetspeak).
2) INSULT/HARASSMENT: direct insulting name-calling (e.g., "you are an idiot", "dickhead", "bastard", etc.).
3) SEXUAL: sexual words, explicit sexual slang, or explicit references to sexual acts/body parts used sexually.

Be as lenient as possible:
- Do NOT flag mild exclamations/interjections as profanity, such as: "oh god", "oh my god", "omg", "gosh", "jeez", "dang", "darn", "heck", "crap" (when not used as a direct insult).
- Do NOT flag normal gameplay violence words like "kill", "die", "shoot" when describing in-game actions.
- Do NOT guess. If you cannot point to an explicit disallowed term, return True.

What to check:
- Scan ALL string fields in the JSON sample.

Output format (STRICT, no backticks/quotes/code blocks):
- If clean: output exactly `True`
- If not clean: output exactly
  `False: <category>; <matched_terms>; <field_path>; <short reason>; <example snippet>`

Rules for False:
- <matched_terms> MUST list the exact offending word(s) found in the sample (verbatim as they appear).
- If you cannot provide matched_terms, you MUST output True.

Allowed categories:
PROFANITY | INSULT | SEXUAL

JSON sample:
{sample}
"""
PROMPT_TEMPLATE = PROMPT_TEMPLATE = """You are a low-risk word-level compliance checker for an open-source gaming dataset.
Your ONLY job is to decide whether the JSON sample contains CLEARLY disallowed words/phrases that could trigger platform review.

Disallowed categories (ONLY these):
1) PROFANITY: strong/explicit swear words or crude vulgar terms, including obfuscated forms (f**k, s#it, f.u.c.k, leetspeak).
2) INSULT: direct harassment/name-calling aimed at a person/group, including identity-based slurs.
3) SEXUAL: explicit sexual words/slang, explicit sexual acts, or explicit sexual body parts used sexually.

Open-source low-risk rules (important):
- Be lenient with mild exclamations/interjections: "oh god", "oh my god", "omg", "gosh", "jeez", "dang", "darn", "heck".
- HOWEVER, for low-risk publishing, DO flag borderline phrases when they are clearly directed at someone or sexualized, even if not the strongest profanity:
  - Directed rudeness/harassment like "shut up" / "stfu" / "get lost" / "screw you" when aimed at a person/name.
  - Sexualized slang/innuendo like "thick boy" / "thicc" when used to describe a person in a sexualized way.
- Do NOT guess. Do NOT infer intent beyond the exact words.
- If you cannot point to an explicit disallowed term/phrase, return True.

Context gating (to avoid over-flagging):
- For INSULT, only flag when it is clearly targeted at someone (e.g., contains "you/your/he/she/they/<name>" near the insult phrase) OR is an identity-based slur.
- Allow self-directed frustration like "Am I an idiot?" unless it contains strong slurs or explicit profanity.

What to check:
- Scan ALL string fields in the JSON sample.
- Only decide based on exact surface words present in the text.

Output format (STRICT, no backticks/quotes/code blocks):
- If clean: output exactly `True`
- If not clean: output exactly
  `False: <category>; <matched_terms>; <field_path>; <short reason>; <example snippet>`

Rules for False:
- <matched_terms> MUST list the exact offending word(s)/phrase(s) found in the sample (verbatim).
- If you cannot provide matched_terms, you MUST output True.

Allowed categories:
PROFANITY | INSULT | SEXUAL

JSON sample:
{sample}
"""

# ========================
# Helpers
# ========================
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def atomic_write_json(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_progress(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
        return int(st.get("last_done", 0))
    except FileNotFoundError:
        return 0
    except Exception:
        return 0

def save_progress(path: str, last_done: int):
    atomic_write_json(path, {"last_done": int(last_done)})

def append_jsonl(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def normalize(resp: str) -> str:
    r = (resp or "").strip()
    while len(r) >= 2 and (r[0] == r[-1]) and r[0] in ("`", '"', "'"):
        r = r[1:-1].strip()
    r = r.splitlines()[0].strip()
    r = re.sub(r"[.。!！]+$", "", r).strip()
    return r

def is_clean(resp: str) -> bool:
    return normalize(resp) == "True"

def parse_false(resp: str) -> Dict[str, str]:
    """
    Parse: False: <category>; <matched_terms>; <field_path>; <short reason>; <example snippet>
    Return dict with best-effort fields.
    """
    r = normalize(resp)
    out = {"category": "UNKNOWN", "matched_terms": "", "field_path": "", "reason": "", "snippet": ""}

    if not r.startswith("False"):
        return out

    # remove leading "False:" if exists
    body = r
    if "False:" in r:
        body = r.split("False:", 1)[1].strip()

    parts = [p.strip() for p in body.split(";")]
    if len(parts) >= 1 and parts[0]:
        out["category"] = parts[0]
    if len(parts) >= 2:
        out["matched_terms"] = parts[1]
    if len(parts) >= 3:
        out["field_path"] = parts[2]
    if len(parts) >= 4:
        out["reason"] = parts[3]
    if len(parts) >= 5:
        out["snippet"] = "; ".join(parts[4:])  # 允许 snippet 里有分号
    return out

def _init_client() -> openai.AzureOpenAI:
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, SCOPE)
    return openai.AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        azure_ad_token_provider=token_provider,
        max_retries=5,
    )

def _check_one(args: Tuple[int, str]) -> Tuple[int, str, Dict[str, Any]]:
    line_no, line = args
    sample_obj = json.loads(line)
    sample_str = json.dumps(sample_obj, ensure_ascii=False)
    prompt = PROMPT_TEMPLATE.format(sample=sample_str)

    global _CLIENT
    if "_CLIENT" not in globals() or _CLIENT is None:
        _CLIENT = _init_client()

    resp = _CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }],
        seed=42,
        temperature=0,
    ).choices[0].message.content or ""

    return line_no, resp.strip(), sample_obj

# ========================
# Main
# ========================
def main():
    workers = 8
    ensure_dir(STATE_DIR)

    last_done = load_progress(PROGRESS_PATH)
    start_line = last_done + 1

    tasks = []
    total_lines = 0
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            total_lines = i
            if i < start_line:
                continue
            line = line.strip()
            if not line:
                # 空行也推进进度，但不送去模型
                tasks.append((i, json.dumps({"_empty_line": True}, ensure_ascii=False)))
                continue
            tasks.append((i, line))

    if not tasks:
        print(f"Nothing to do. last_done={last_done}, total_lines={total_lines}")
        return

    print(f"Resume from line {start_line} (last_done={last_done}), pending={len(tasks)}, total_lines≈{total_lines}")

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=workers, maxtasksperchild=200)

    done_set = set()
    next_expected = start_line

    # stats
    checked = 0
    clean_cnt = 0
    blocked_cnt = 0
    cat_counter = Counter()
    example_by_cat = defaultdict(list)  # 只留少量例子

    try:
        for line_no, resp, sample_obj in pool.imap_unordered(_check_one, tasks, chunksize=8):
            checked += 1
            norm = normalize(resp)
            print(f"[{line_no}] {norm}")

            done_set.add(line_no)

            progressed = False
            while next_expected in done_set:
                done_set.remove(next_expected)
                last_done = next_expected
                next_expected += 1
                progressed = True

            if progressed:
                save_progress(PROGRESS_PATH, last_done)

            if is_clean(resp):
                clean_cnt += 1
            else:
                blocked_cnt += 1
                info = parse_false(resp)
                cat = info.get("category", "UNKNOWN") or "UNKNOWN"
                cat_counter[cat] += 1

                # 写入 blocked.jsonl
                append_jsonl(BLOCKED_PATH, {
                    "line_no": line_no,
                    "resp": norm,
                    "parsed": info,
                    "sample": sample_obj,
                })

                # 记录少量例子用于最终展示
                if len(example_by_cat[cat]) < 3:
                    example_by_cat[cat].append({
                        "line_no": line_no,
                        "matched_terms": info.get("matched_terms", ""),
                        "snippet": info.get("snippet", ""),
                    })

        pool.close()
        pool.join()
        save_progress(PROGRESS_PATH, last_done)

    except Exception:
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass
        raise

    # ========================
    # Final summary
    # ========================
    print("\n====== Safety Check Summary ======")
    print(f"File: {FILE_PATH}")
    print(f"Progress last_done: {last_done}  (next start: {last_done + 1})")
    print(f"Checked (this run): {checked}")
    print(f"Clean:   {clean_cnt}")
    print(f"Blocked: {blocked_cnt}")
    print(f"Blocked file: {BLOCKED_PATH}")
    print("Category counts:")
    if not cat_counter:
        print("  (none)")
    else:
        for k, v in cat_counter.most_common():
            print(f"  - {k}: {v}")

    if example_by_cat:
        print("\nExamples (up to 3 each):")
        for cat, exs in example_by_cat.items():
            print(f"  [{cat}]")
            for ex in exs:
                print(f"    - line {ex['line_no']}: terms={ex['matched_terms']} | snippet={ex['snippet']}")

if __name__ == "__main__":
    main()
