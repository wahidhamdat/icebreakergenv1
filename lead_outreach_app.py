"""
Lead Outreach Pipeline: Validate -> Segment -> Scrape -> Icebreak.
Supabase tables for leads, ice output, and rejected. Full control over API inputs.
"""
import os
import re
import json
import time
from datetime import datetime, timezone
from urllib.parse import quote
import streamlit as st
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Default prompts (include li_posts and segment)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """You write short, casual email icebreakers for logistics professionals. Sound like a colleague, not a salesperson. Reference their company and city naturally. Keep under 25 words. Never say 'I noticed' or 'I saw'. If they have recent posts, mention them subtly. Return ONLY the icebreaker text."""

DEFAULT_USER_TEMPLATE = """Write one icebreaker for this person.

First Name: {first_name}
Title: {title}
Headline: {headline}
Company: {company}
City: {city}
Country: {country}
Segment: {segment}
Recent LinkedIn posts: {li_posts}

Pick the single most specific detail. Return only the icebreaker, starting with Hey."""

# Column aliases (same as main app)
COLUMN_ALIASES = {
    "first_name": ["first_name", "firstName", "First Name"],
    "last_name": ["last_name", "lastName", "Last Name"],
    "full_name": ["full_name", "fullName", "Full Name", "name", "Name"],
    "email": ["email", "Email"],
    "title": ["title", "Title", "job_title", "Job Title"],
    "headline": ["headline", "Headline"],
    "company": ["company", "Company", "company_name", "Company Name"],
    "linkedin_url": ["linkedin_url", "linkedin", "LinkedIn"],
    "city": ["city", "City"],
    "country": ["country", "Country"],
    "industry": ["industry", "Industry"],
    "website": ["website", "company_website", "Company Website", "url"],
    "employees": ["employees", "employees_count", "Employees Count"],
    "keywords": ["keywords", "Keywords"],
}


def normalize_contact_row(raw: dict) -> dict:
    """Map various column names to standard keys. Require email."""
    result = {}
    raw_lower = {str(k).strip().lower(): k for k in raw.keys()}
    for standard_key, aliases in COLUMN_ALIASES.items():
        value = None
        for alias in aliases:
            key_lower = alias.lower()
            if key_lower in raw_lower:
                orig_key = raw_lower[key_lower]
                v = raw.get(orig_key)
                if v is not None and str(v).strip() != "":
                    value = str(v).strip()
                    break
        result[standard_key] = value or ""
    if not result.get("full_name") and (result.get("first_name") or result.get("last_name")):
        result["full_name"] = f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
    if "company_website" in raw and not result.get("website"):
        result["website"] = str(raw.get("company_website", "") or "").strip()
    if "employees_count" in raw and result.get("employees") == "":
        result["employees"] = raw.get("employees_count") if raw.get("employees_count") is not None else ""
    return result


def clean_and_dedup(rows: list[dict]) -> list[dict]:
    """Dedup by email (lowercase), require email present."""
    seen = set()
    out = []
    for r in rows:
        email = (r.get("email") or "").strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        out.append(r)
    return out


def parse_csv_to_rows(uploaded_file) -> list[dict]:
    df = pd.read_csv(uploaded_file)
    return [normalize_contact_row(r.to_dict()) for _, r in df.iterrows() if normalize_contact_row(r.to_dict()).get("email")]


def fetch_url_to_rows(url: str) -> list[dict]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content = resp.content
    try:
        data = resp.json()
        items = data if isinstance(data, list) else (data.get("data") or data.get("rows") or data.get("contacts") or [])
        if not isinstance(items, list):
            items = [data]
        return [normalize_contact_row(i) for i in items if isinstance(i, dict) and normalize_contact_row(i).get("email")]
    except Exception:
        pass
    import io
    df = pd.read_csv(io.BytesIO(content))
    return [normalize_contact_row(r.to_dict()) for _, r in df.iterrows() if normalize_contact_row(r.to_dict()).get("email")]


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
def supabase_headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def load_leads_table(url: str, key: str, table_name: str, limit: int = 2000) -> list[dict]:
    resp = requests.get(
        f"{url}/rest/v1/{table_name}",
        params={"select": "*", "email": "not.is.null", "limit": str(limit)},
        headers=supabase_headers(key),
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    return [normalize_contact_row(r) for r in rows]


def upsert_ice_row(url: str, key: str, table_name: str, row: dict, icebreaker: str, on_conflict: str = "email") -> None:
    headers = supabase_headers(key)
    headers["Prefer"] = "resolution=merge-duplicates"
    body = {
        "ai_icebreaker": icebreaker,
        "first_name": row.get("first_name", ""),
        "last_name": row.get("last_name", ""),
        "full_name": row.get("full_name", ""),
        "email": row.get("email", ""),
        "title": row.get("title", ""),
        "headline": row.get("headline", ""),
        "city": row.get("city", ""),
        "country": row.get("country", ""),
        "linkedin_url": row.get("linkedin_url", ""),
        "company": row.get("company", ""),
        "website": row.get("website", ""),
        "industry": row.get("industry", ""),
        "employees": str(row.get("employees", "")),
        "keywords": row.get("keywords", ""),
        "segment": row.get("segment", ""),
        "li_posts": row.get("li_posts", ""),
    }
    resp = requests.post(f"{url}/rest/v1/{table_name}?on_conflict={on_conflict}", json=body, headers=headers, timeout=30)
    if resp.status_code == 409:
        email = row.get("email", "")
        if email:
            requests.patch(
                f"{url}/rest/v1/{table_name}?email=eq.{quote(email, safe='')}",
                json=body,
                headers=supabase_headers(key),
                timeout=30,
            ).raise_for_status()
        else:
            resp.raise_for_status()
    else:
        resp.raise_for_status()


def insert_rejected_batch(url: str, key: str, table_name: str, rows: list[dict]) -> None:
    """Insert rejected leads into the rejected table (one by one to avoid payload size limits)."""
    for r in rows:
        body = {
            "full_name": r.get("full_name", ""),
            "email": r.get("email", ""),
            "title": r.get("title", ""),
            "company": r.get("company", ""),
            "validation_status": r.get("validation_status", ""),
            "rejection_reason": r.get("rejection_reason", ""),
            "typo_suggestion": r.get("typo_suggestion", ""),
        }
        try:
            requests.post(
                f"{url}/rest/v1/{table_name}",
                json=body,
                headers=supabase_headers(key),
                timeout=30,
            ).raise_for_status()
        except Exception:
            pass  # best-effort; user can still download CSV


# ---------------------------------------------------------------------------
# Validation (free batch API)
# ---------------------------------------------------------------------------
def validate_emails_batch(api_url: str, emails: list[str], timeout: int = 30) -> list[dict]:
    """POST { emails: [...] } to validation API. Returns list of per-email results."""
    if not emails:
        return []
    resp = requests.post(api_url, json={"emails": emails}, headers={"Content-Type": "application/json"}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    return results if isinstance(results, list) else []


def merge_validation_with_leads(leads: list[dict], batch_results: list[dict]) -> list[dict]:
    """Merge validation API results with lead data. Add validation_status, is_valid, rejection_reason."""
    email_to_lead = {(r.get("email") or "").strip().lower(): r for r in leads}
    validated_emails = set()
    out = []
    for r in batch_results:
        email = (r.get("email") or "").strip().lower()
        lead = email_to_lead.get(email) or {}
        status = r.get("status") or "UNKNOWN"
        is_valid = status in ("VALID", "PROBABLY_VALID")
        validations = r.get("validations") or {}
        reason = "Unknown"
        if status == "INVALID_FORMAT":
            reason = "Invalid email format (syntax)"
        elif status == "INVALID_DOMAIN":
            reason = "Domain does not exist"
        elif status == "DISPOSABLE":
            reason = "Disposable/temp email"
        elif validations.get("is_disposable"):
            reason = "Disposable email detected"
        elif not validations.get("mx_records"):
            reason = "No MX record"
        elif not validations.get("domain_exists"):
            reason = "Domain does not exist"
        elif status == "API_ERROR":
            reason = "Email validator API error"
        else:
            reason = f"Failed validation ({status})"
        typo = r.get("typoSuggestion") or r.get("typo_suggestion") or ""
        if typo:
            reason += f" → Did you mean: {typo}?"
        validated_emails.add(email)
        out.append({
            **lead,
            "email": email or lead.get("email", ""),
            "validation_status": status,
            "is_valid": is_valid,
            "syntax_ok": validations.get("syntax", False),
            "domain_exists": validations.get("domain_exists", False),
            "has_mx": validations.get("mx_records", False),
            "rejection_reason": reason,
            "typo_suggestion": typo,
        })
    for email, lead in email_to_lead.items():
        if email and email not in validated_emails:
            out.append({
                **lead,
                "validation_status": "API_ERROR",
                "is_valid": False,
                "rejection_reason": "Email validator API error or missing",
                "typo_suggestion": "",
            })
    return out


# ---------------------------------------------------------------------------
# Segment (regex on title + headline)
# ---------------------------------------------------------------------------
def segment_lead(lead: dict, rules: list[tuple[str, str]]) -> dict:
    """Apply regex rules in order; first match sets segment. Default Unsegmented."""
    title = (lead.get("title") or "").strip()
    headline = (lead.get("headline") or "").strip()
    combined = f"{title} | {headline}"
    segment = "Unsegmented"
    for pattern, label in rules:
        try:
            if re.search(pattern, combined, re.IGNORECASE):
                segment = label
                break
        except re.error:
            continue
    return {**lead, "segment": segment}


# ---------------------------------------------------------------------------
# LinkedIn username + Apify scrape
# ---------------------------------------------------------------------------
def get_linkedin_username(linkedin_url: str) -> str:
    if not linkedin_url or not str(linkedin_url).strip():
        return ""
    m = re.search(r"linkedin\.com/in/([^/?]+)", str(linkedin_url), re.IGNORECASE)
    return m.group(1).strip() if m else ""


def apify_run_sync(token: str, actor_id: str, run_input: dict, timeout: int = 120) -> list[dict]:
    """Run Apify actor sync and return dataset items. actor_id can be 'user~actor' or 'user/actor'."""
    act = actor_id.replace("/", "~")
    url = f"https://api.apify.com/v2/acts/{act}/run-sync-get-dataset-items?token={token}"
    resp = requests.post(url, json=run_input, headers={"Content-Type": "application/json"}, timeout=timeout)
    resp.raise_for_status()
    return resp.json() if resp.content else []


def parse_li_posts(items: list[dict], max_posts: int = 2, max_chars: int = 200) -> str:
    """Extract post text from Apify LinkedIn posts response. Join with ' | '."""
    parts = []
    for post in (items or [])[:max_posts]:
        text = (post.get("text") or post.get("content") or post.get("description") or "").strip()
        if len(text) > 30:
            clean = re.sub(r"\s+\n\s+", " ", text).replace("\n", " ").strip()[:max_chars]
            if len(clean) > 10:
                parts.append(clean)
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# AI icebreaker (OpenAI-compatible API)
# ---------------------------------------------------------------------------
def build_user_prompt(row: dict, template: str) -> str:
    placeholders = ["first_name", "last_name", "full_name", "email", "title", "headline", "company", "city", "country", "industry", "employees", "website", "keywords", "segment", "li_posts"]
    out = template
    for key in placeholders:
        val = row.get(key)
        s = str(val).strip() if val else ""
        out = out.replace("{" + key + "}", s)
    return out


def _clean_icebreaker(raw: str) -> str:
    raw = re.sub(r"^Icebreaker\s*:?\s*", "", str(raw).strip(), flags=re.IGNORECASE).strip("'\"")
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    hey = next((l for l in lines if l.startswith("Hey")), None)
    return hey or (lines[-1] if lines else raw)


def generate_icebreaker(row: dict, api_url: str, api_key: str, model: str, system_prompt: str, user_template: str, timeout: int = 60) -> str:
    user_content = build_user_prompt(row, user_template)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }
    resp = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("choices"):
        return "[Draft manually]"
    msg = data["choices"][0].get("message", {})
    content = (msg.get("content") or "").strip()
    if not content or len(content) < 10:
        return "[Draft manually]"
    return _clean_icebreaker(content)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Lead Outreach Pipeline", layout="wide")
st.title("Lead Outreach: Validate → Segment → Scrape → Icebreak")
st.caption("Load leads from Supabase/CSV/URL → validate (free API) → segment → scrape LinkedIn (Apify) → generate icebreaker → save to ice table. Rejected leads stored separately; download only.")

# Session state
if "outreach_leads" not in st.session_state:
    st.session_state["outreach_leads"] = None
if "valid_leads" not in st.session_state:
    st.session_state["valid_leads"] = []
if "rejected_leads" not in st.session_state:
    st.session_state["rejected_leads"] = []
if "pipeline_done" not in st.session_state:
    st.session_state["pipeline_done"] = False
if "stop_requested" not in st.session_state:
    st.session_state["stop_requested"] = False
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False

# Sidebar: full control
with st.sidebar:
    st.header("Supabase")
    supabase_url = st.text_input("Supabase URL", value=os.environ.get("SUPABASE_URL", ""), help="Project URL")
    supabase_key = st.text_input("Supabase Anon Key", type="password", value=os.environ.get("SUPABASE_ANON_KEY", ""))
    leads_table = st.text_input("Leads table (source)", value=os.environ.get("LEADS_TABLE_NAME", "contacts"))
    ice_table = st.text_input("Ice table (destination)", value=os.environ.get("ICE_TABLE_NAME", "ice"))
    rejected_table = st.text_input("Rejected table", value=os.environ.get("REJECTED_TABLE_NAME", "rejected"), help="Needs columns: email, full_name, title, company, validation_status, rejection_reason, typo_suggestion, rejected_at")
    on_conflict = st.text_input("Upsert conflict column", value="email")

    st.divider()
    st.header("Validation API (free)")
    validation_api_url = st.text_input(
        "Validation API URL",
        value=os.environ.get("VALIDATION_API_URL", "https://rapid-email-verifier.fly.dev/api/validate/batch"),
    )
    validation_timeout = st.number_input("Validation timeout (s)", min_value=10, max_value=120, value=30)

    st.divider()
    st.header("Apify (LinkedIn scrape)")
    apify_token = st.text_input("Apify API token", type="password", value=os.environ.get("APIFY_TOKEN", ""))
    apify_actor_id = st.text_input(
        "Actor ID",
        value=os.environ.get("APIFY_ACTOR_ID", "datadoping~linkedin-profile-posts-scraper"),
        help="e.g. datadoping~linkedin-profile-posts-scraper",
    )
    apify_input_json = st.text_area(
        "Run input (JSON). Use {{username}} for LinkedIn username.",
        value='{"profiles": ["{{username}}"], "max_posts": 3, "max_comments_per_post": 0}',
        height=100,
    )
    apify_timeout = st.number_input("Apify timeout (s)", min_value=30, max_value=300, value=90)
    scrape_delay = st.number_input("Delay between scrapes (s)", min_value=0, max_value=10, value=1)

    st.divider()
    st.header("AI (Icebreaker)")
    ai_api_url = st.text_input(
        "AI API URL",
        value=os.environ.get("AI_API_URL", "https://api.deepseek.com/v1/chat/completions"),
        help="OpenAI-compatible (DeepSeek, Cerebras, etc.)",
    )
    ai_api_key = st.text_input("AI API Key", type="password", value=os.environ.get("AI_API_KEY", ""))
    ai_model = st.text_input("Model", value=os.environ.get("AI_MODEL", "deepseek-chat"))
    with st.expander("Edit AI prompts"):
        system_prompt = st.text_area("System prompt", value=st.session_state.get("outreach_system_prompt", DEFAULT_SYSTEM_PROMPT), height=120)
        user_template = st.text_area("User template", value=st.session_state.get("outreach_user_template", DEFAULT_USER_TEMPLATE), height=120)
        st.session_state["outreach_system_prompt"] = system_prompt
        st.session_state["outreach_user_template"] = user_template

    st.divider()
    st.header("Segment rules (regex => label)")
    segment_rules_text = st.text_area(
        "One per line: regex => Label",
        value="\\b(founder|owner|co-?founder|partner|proprietor)\\b => Founders and Owners\n"
              "\\b(general\\s*manager|\\bgm\\b|managing\\s*director|country\\s*manager)\\b => General Directors\n"
              "\\b(operations?|ops)\\s*(manager|director|lead|head|vp|chief)\\b|\\b(manager|director)\\s*(of\\s+)?(operations?|ops)\\b => Operations Managers",
        height=100,
    )

supabase_ready = bool(supabase_url and supabase_key)
if not supabase_ready:
    st.info("Enter Supabase URL and Anon Key in the sidebar.")
    st.stop()

# Parse segment rules
def _parse_segment_rules(text: str) -> list[tuple[str, str]]:
    rules = []
    for line in (segment_rules_text or "").strip().split("\n"):
        line = line.strip()
        if "=>" in line:
            a, b = line.split("=>", 1)
            rules.append((a.strip(), b.strip()))
    if not rules:
        rules = [(r"\b(operations?|ops)\s*(manager|director)\b", "Operations Managers"), (r"\b(founder|owner)\b", "Founders and Owners")]
    return rules

# Tabs
tab_load, tab_run, tab_results = st.tabs(["Load", "Run pipeline", "Results"])

with tab_load:
    load_source = st.radio("Source", ["Supabase (leads table)", "Upload CSV", "Import from URL"], horizontal=True)
    if load_source == "Supabase (leads table)":
        if st.button("Load from Supabase", type="primary"):
            with st.spinner("Loading..."):
                try:
                    rows = load_leads_table(supabase_url, supabase_key, leads_table)
                    rows = clean_and_dedup(rows)
                    st.session_state["outreach_leads"] = rows
                    st.success(f"Loaded **{len(rows)}** leads.")
                except Exception as e:
                    st.error(str(e))
    elif load_source == "Upload CSV":
        up = st.file_uploader("CSV", type=["csv"])
        if up:
            try:
                rows = parse_csv_to_rows(up)
                rows = clean_and_dedup(rows)
                st.session_state["outreach_leads"] = rows
                st.success(f"Loaded **{len(rows)}** leads.")
            except Exception as e:
                st.error(str(e))
    else:
        url_in = st.text_input("URL (JSON or CSV)")
        if st.button("Fetch"):
            if url_in.strip():
                try:
                    rows = fetch_url_to_rows(url_in.strip())
                    rows = clean_and_dedup(rows)
                    st.session_state["outreach_leads"] = rows
                    st.success(f"Loaded **{len(rows)}** leads.")
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("outreach_leads"):
        df = pd.DataFrame(st.session_state["outreach_leads"])
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)

with tab_run:
    leads = st.session_state.get("outreach_leads") or []
    if not leads:
        st.warning("Load leads in the **Load** tab first.")
    else:
        st.metric("Leads loaded", len(leads))
        run_btn = st.button("Run: Validate → Segment → Scrape → Icebreak", type="primary", use_container_width=True)
        if run_btn:
            st.session_state["is_running"] = True
            st.session_state["stop_requested"] = False
            st.session_state["valid_leads"] = []
            st.session_state["rejected_leads"] = []
            progress = st.progress(0)
            status_ph = st.empty()

            try:
                # Clean & dedup
                leads = clean_and_dedup(leads)
                # Validate in chunks of 50
                chunk_size = 50
                all_validated = []
                for i in range(0, len(leads), chunk_size):
                    chunk = leads[i : i + chunk_size]
                    emails = [r.get("email", "").strip().lower() for r in chunk if r.get("email")]
                    if not emails:
                        continue
                    status_ph.caption(f"Validating batch {i // chunk_size + 1}...")
                    try:
                        results = validate_emails_batch(validation_api_url, emails, validation_timeout)
                        merged = merge_validation_with_leads(chunk, results)
                        all_validated.extend(merged)
                    except Exception as e:
                        for r in chunk:
                            all_validated.append({**r, "validation_status": "API_ERROR", "is_valid": False, "rejection_reason": str(e), "typo_suggestion": ""})
                    progress.progress(min(1.0, (i + chunk_size) / max(len(leads), 1)))

                valid = [r for r in all_validated if r.get("is_valid")]
                rejected = [r for r in all_validated if not r.get("is_valid")]

                # Tag rejected for storage
                for r in rejected:
                    if "rejection_reason" not in r:
                        r["rejection_reason"] = r.get("validation_status", "Unknown")
                    r["typo_suggestion"] = r.get("typo_suggestion") or ""

                st.session_state["rejected_leads"] = rejected
                status_ph.caption(f"Valid: {len(valid)}, Rejected: {len(rejected)}. Saving rejected to DB...")

                if rejected and supabase_ready:
                    insert_rejected_batch(supabase_url, supabase_key, rejected_table, rejected)

                # Segment valid
                rules = _parse_segment_rules(segment_rules_text)
                valid = [segment_lead(r, rules) for r in valid]

                if not valid:
                    status_ph.caption("No valid leads to process.")
                    st.session_state["pipeline_done"] = True
                    st.session_state["is_running"] = False
                    st.stop()

                # Apify input: replace {{username}}
                try:
                    apify_input = json.loads(apify_input_json)
                except Exception:
                    apify_input = {"profiles": ["{{username}}"], "max_posts": 3}

                total = len(valid)
                saved = 0
                for idx, lead in enumerate(valid):
                    if st.session_state.get("stop_requested"):
                        status_ph.caption("Stopped by user.")
                        break
                    progress.progress((idx + 1) / total)
                    status_ph.caption(f"Processing {idx + 1}/{total}: {lead.get('email', '')}")

                    username = get_linkedin_username(lead.get("linkedin_url") or "")
                    li_posts = ""
                    if username and apify_token:
                        try:
                            inp = json.loads(json.dumps(apify_input))
                            def replace_username(obj):
                                if isinstance(obj, dict):
                                    return {k: replace_username(v) for k, v in obj.items()}
                                if isinstance(obj, list):
                                    return [replace_username(x) for x in obj]
                                if isinstance(obj, str):
                                    return obj.replace("{{username}}", username)
                                return obj
                            inp = replace_username(inp)
                            items = apify_run_sync(apify_token, apify_actor_id, inp, apify_timeout)
                            li_posts = parse_li_posts(items)
                        except Exception:
                            pass
                        time.sleep(scrape_delay)
                    lead = {**lead, "li_posts": li_posts or "(none)"}

                    if ai_api_key:
                        try:
                            _sys = st.session_state.get("outreach_system_prompt", DEFAULT_SYSTEM_PROMPT)
                            _tpl = st.session_state.get("outreach_user_template", DEFAULT_USER_TEMPLATE)
                            ice = generate_icebreaker(lead, ai_api_url, ai_api_key, ai_model, _sys, _tpl)
                        except Exception as e:
                            ice = f"[Error: {e}]"
                    else:
                        ice = "[Set AI API Key]"

                    lead["ai_icebreaker"] = ice
                    try:
                        upsert_ice_row(supabase_url, supabase_key, ice_table, lead, ice, on_conflict)
                        saved += 1
                    except Exception:
                        pass

                st.session_state["valid_leads"] = valid
                st.session_state["pipeline_done"] = True
                status_ph.caption(f"Done. Saved {saved} to ice table.")
                st.success(f"Saved **{saved}** leads to ice table. Rejected: **{len(rejected)}** (see Results tab).")
            except Exception as e:
                st.error(str(e))
            finally:
                st.session_state["is_running"] = False

        if st.session_state.get("is_running"):
            if st.button("Stop"):
                st.session_state["stop_requested"] = True

with tab_results:
    st.subheader("Rejected / failed leads")
    rejected = st.session_state.get("rejected_leads") or []
    if rejected:
        st.caption("These leads did not pass validation. They are not in the ice table. Download if you need them.")
        df_rej = pd.DataFrame(rejected)
        display_cols = [c for c in ["email", "full_name", "company", "title", "validation_status", "rejection_reason", "typo_suggestion"] if c in df_rej.columns]
        if display_cols:
            st.dataframe(df_rej[display_cols].fillna(""), use_container_width=True, hide_index=True, height=300)
        st.download_button(
            "Download rejected leads (CSV)",
            df_rej.to_csv(index=False),
            file_name="rejected_leads.csv",
            mime="text/csv",
            key="dl_rejected",
        )
    else:
        st.info("No rejected leads, or run the pipeline first.")

    st.subheader("Valid / saved")
    valid = st.session_state.get("valid_leads") or []
    if valid:
        st.caption(f"Saved **{len(valid)}** leads to the ice table. Open your Supabase project to view the **{ice_table}** table.")
        st.dataframe(pd.DataFrame(valid).head(20), use_container_width=True, hide_index=True)
    else:
        st.caption("Run the pipeline to see saved leads here.")
