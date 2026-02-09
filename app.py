import os
from urllib.parse import quote
import streamlit as st
import requests
import time
import re
import pandas as pd

# ---------------------------------------------------------------------------
# System prompt (identical to n8n workflow)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a freight industry insider writing cold email icebreakers for outreach to small U.S. freight brokerages and 3PLs (1-20 employees).

YOUR PROCESS:
1. THINK about which single detail from the lead's data is most specific and personal. Rank: headline insight > keyword clue > city+company combo > generic title.
2. SHORTEN the company name: drop Inc, LLC, Corp, Group, Solutions, Services, International, Global, Holdings, Partners, Associates, Enterprises. If still 3+ words, use first 1-2 distinctive words.
3. SHORTEN cities: Los Angeles=LA, Dallas=DFW, Chicago=CHI, Atlanta=ATL, San Francisco=SF, New York=NYC, Indianapolis=Indy, Philadelphia=Philly, Jacksonville=Jax, Salt Lake City=SLC, Kansas City=KC, St. Louis=STL, Phoenix=PHX, Charlotte=CLT, Nashville=Nash, Memphis=MEM, Tampa=TPA, Denver=DEN, Portland=PDX, Seattle=SEA
4. WRITE the icebreaker.

FORMAT: Hey [First Name], [personal detail that proves homework]. [Open-ended curiosity hook].
Max 25 words. Max 2 sentences. Start with Hey. No exclamation marks. No emojis. No questions. No pitch.
Use freight language naturally: lanes, loads, brokerage, carriers, capacity game, quoting grind, carrier mix

NEVER SAY: "I came across", "I found you", "I noticed", "I saw", "Hope this finds you", "reaching out because", "I wanted to introduce", "my name is", "innovative", "cutting-edge", "streamline", "leverage", "synergy", "solution", "platform", "We help", "We offer"

EXAMPLES:
- Jake, CEO, Houston, Apex Freight Solutions Inc
  > Hey Jake, love what Apex is building out of Houston. Also in freight -- wanted to run something by you.

- Trent, Ops Manager, Dallas, Velocity Logistics Group LLC
  > Hey Trent, the multi-carrier quoting game at Velocity's volume is no joke. Had a thought worth 2 min.

- Maria, President, El Paso, Puente Logistics International
  > Hey Maria, respect what Puente is doing in cross-border out of El Paso. Quick thought for you.

- Keisha, Dispatch Manager, Savannah, Harbor Freight Logistics LLC
  > Hey Keisha, drayage ops out of Savannah with that port volume takes real chops. Had an idea for Harbor.

- Omar, Dir of Ops, Atlanta, ABGL Freight Services
  > Hey Omar, running reefer ops out of ATL at ABGL's pace is a different animal. Wanted to run something by you.

- Danielle, VP Operations, Memphis, Velocity Supply Chain Solutions
  > Hey Danielle, scaling a 3PL out of MEM without adding headcount is the real puzzle. Had a thought for Velocity.

- Derek, Founder & CEO, Nashville, Pinnacle Transport Holdings Inc
  > Hey Derek, building a brokerage from scratch in Nash takes guts. Respect what Pinnacle has going.

OUTPUT: Return ONLY the icebreaker text. No quotes. No labels. No explanation. First character must be H."""

DEFAULT_USER_TEMPLATE = """Write one icebreaker for this person. Pick the single most specific detail.

First Name: {first_name}
Title: {title}
Headline: {headline}
Seniority: {seniority}
Department: {department}
City: {city}
Country: {country}
Company: {company}
Industry: {industry}
Employees: {employees}
Website: {website}
Keywords: {keywords}"""

# Column name aliases for CSV/URL imports
COLUMN_ALIASES = {
    "first_name": ["first_name", "firstName", "first name", "First Name"],
    "last_name": ["last_name", "lastName", "last name", "Last Name"],
    "full_name": ["full_name", "fullName", "full name", "Full Name", "name", "Name"],
    "email": ["email", "Email", "email_address", "Email Address"],
    "title": ["title", "Title", "job_title", "Job Title"],
    "headline": ["headline", "Headline"],
    "seniority": ["seniority", "Seniority"],
    "department": ["department", "Department"],
    "city": ["city", "City"],
    "country": ["country", "Country"],
    "linkedin_url": ["linkedin_url", "linkedin", "LinkedIn", "linkedin_url"],
    "company": ["company", "Company", "company_name", "Company Name", "organization"],
    "website": ["website", "Website", "company_website", "url"],
    "company_linkedin": ["company_linkedin", "Company LinkedIn"],
    "industry": ["industry", "Industry"],
    "employees": ["employees", "Employees", "employees_count", "employee_count"],
    "keywords": ["keywords", "Keywords"],
}


def normalize_contact_row(raw: dict) -> dict:
    """Map various column names to the keys expected by build_user_prompt and upsert_to_ice."""
    result = {}
    for standard_key, aliases in COLUMN_ALIASES.items():
        value = None
        raw_lower = {str(k).strip().lower(): k for k in raw.keys()}
        for alias in aliases:
            key_lower = alias.lower()
            if key_lower in raw_lower:
                orig_key = raw_lower[key_lower]
                v = raw.get(orig_key)
                if v is not None and str(v).strip() != "":
                    value = str(v).strip()
                    break
        result[standard_key] = value or ""
    # full_name fallback
    if not result.get("full_name") and (result.get("first_name") or result.get("last_name")):
        result["full_name"] = f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
    return result


def parse_csv_to_rows(uploaded_file) -> list[dict]:
    """Parse uploaded CSV to list of normalized contact dicts (rows with email only)."""
    df = pd.read_csv(uploaded_file)
    rows = []
    for _, r in df.iterrows():
        raw = r.to_dict()
        normalized = normalize_contact_row(raw)
        if normalized.get("email"):
            rows.append(normalized)
    return rows


def fetch_url_to_rows(url: str) -> list[dict]:
    """Fetch URL; parse JSON or CSV to list of normalized contact dicts."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content = resp.content
    try:
        # Try JSON first
        data = resp.json()
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("data") or data.get("rows") or data.get("contacts") or data.get("results") or []
            if not isinstance(items, list):
                items = [data]
        else:
            items = []
        rows = [normalize_contact_row(item) for item in items if isinstance(item, dict)]
        return [r for r in rows if r.get("email")]
    except Exception:
        pass
    # CSV
    import io
    df = pd.read_csv(io.BytesIO(content))
    rows = []
    for _, r in df.iterrows():
        normalized = normalize_contact_row(r.to_dict())
        if normalized.get("email"):
            rows.append(normalized)
    return rows


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
def supabase_headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def load_ice_table(url: str, key: str, table_name: str, limit: int = 2000) -> list[dict]:
    """Load the full ice table from Supabase."""
    resp = requests.get(
        f"{url}/rest/v1/{table_name}",
        params={"select": "*", "limit": str(limit)},
        headers=supabase_headers(key),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_ice_row_by_email(url: str, key: str, table_name: str, email: str):
    """Fetch one ice table row by email. Returns the row dict or None if not found."""
    email = (email or "").strip()
    if not email:
        return None
    resp = requests.get(
        f"{url}/rest/v1/{table_name}",
        params={"select": "email,ai_icebreaker", "email": f"eq.{quote(email, safe='')}"},
        headers=supabase_headers(key),
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    return rows[0] if rows else None


def has_valid_icebreaker(row) -> bool:
    """True if row has a non-empty ai_icebreaker that is not an error placeholder."""
    if not row or not isinstance(row, dict):
        return False
    val = row.get("ai_icebreaker")
    if val is None or not str(val).strip():
        return False
    s = str(val).strip()
    if s.startswith(("ERROR", "API_ERROR", "REASONING_EXHAUSTED")):
        return False
    return True


def _normalize_contact_row(r: dict) -> dict:
    """Normalize contacts table row so build_user_prompt and upsert_to_ice see consistent keys."""
    out = dict(r)
    if "company_website" in out and "website" not in out:
        out["website"] = out.get("company_website") or ""
    if "employees_count" in out:
        out["employees"] = out.get("employees_count") if out.get("employees_count") is not None else out.get("employees", "")
    return out


def load_contacts(url: str, key: str, contacts_table_name: str, limit: int = 2000) -> list[dict]:
    """Load contacts that have an email from the given table. Rows normalized for prompts/upsert."""
    resp = requests.get(
        f"{url}/rest/v1/{contacts_table_name}",
        params={"select": "*", "email": "not.is.null", "limit": str(limit)},
        headers=supabase_headers(key),
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    return [_normalize_contact_row(r) for r in rows]


def upsert_to_ice(url: str, key: str, table_name: str, row: dict, icebreaker: str, on_conflict: str = "email") -> None:
    """UPSERT a row into the ice table with the generated icebreaker. Uses on_conflict for merge; on 409 retries with PATCH."""
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
        "seniority": row.get("seniority", ""),
        "department": row.get("department", ""),
        "city": row.get("city", ""),
        "country": row.get("country", ""),
        "linkedin_url": row.get("linkedin_url", ""),
        "company": row.get("company", ""),
        "website": row.get("website", ""),
        "company_linkedin": row.get("company_linkedin", ""),
        "industry": row.get("industry", ""),
        "employees": row.get("employees", ""),
        "keywords": row.get("keywords", ""),
    }

    resp = requests.post(f"{url}/rest/v1/{table_name}?on_conflict={on_conflict}", json=body, headers=headers, timeout=30)
    if resp.status_code == 409:
        # Fallback: PATCH by email (e.g. RLS or conflict target mismatch)
        email = row.get("email", "")
        if email:
            patch_resp = requests.patch(
                f"{url}/rest/v1/{table_name}?email=eq.{quote(email, safe='')}",
                json=body,
                headers=supabase_headers(key),
                timeout=30,
            )
            patch_resp.raise_for_status()
        else:
            resp.raise_for_status()
    else:
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Cerebras API
# ---------------------------------------------------------------------------
def build_user_prompt(row: dict) -> str:
    """Build the per-contact user prompt from template and row (safe placeholder replace)."""
    template = st.session_state.get("user_prompt_template", DEFAULT_USER_TEMPLATE)
    # Safe replace: only fill known placeholders so missing keys don't break
    placeholders = ["first_name", "last_name", "full_name", "email", "title", "headline", "seniority", "department", "city", "country", "linkedin_url", "company", "website", "company_linkedin", "industry", "employees", "keywords"]
    out = template
    for key in placeholders:
        val = row.get(key)
        s = str(val).strip() if val else ""
        out = out.replace("{" + key + "}", s)
    return out


def generate_icebreaker(cerebras_key: str, row: dict) -> str:
    """Call Cerebras API and return the icebreaker text."""
    user_prompt = build_user_prompt(row)
    system_prompt = st.session_state.get("system_prompt", SYSTEM_PROMPT)

    resp = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {cerebras_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-oss-120b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.5,
            "max_completion_tokens": 4096,
            "reasoning_effort": "low",
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    # Extract icebreaker from response
    if not data.get("choices"):
        return "ERROR: No choices in API response"

    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content")

    # Primary: use content field
    if content and str(content).strip():
        raw = str(content).strip()
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        if raw:
            return _clean_icebreaker(raw)

    # Fallback: check reasoning field
    reasoning = msg.get("reasoning")
    if reasoning:
        lines = [l.strip() for l in str(reasoning).split("\n") if l.strip()]
        hey_line = next((l for l in lines if l.startswith("Hey")), None)
        if hey_line:
            return _clean_icebreaker(hey_line)

    finish_reason = choice.get("finish_reason", "unknown")
    return f"REASONING_EXHAUSTED: finish={finish_reason}"


def _clean_icebreaker(raw: str) -> str:
    """Extract and clean the icebreaker from raw LLM output."""
    match = re.search(r"ICEBREAKER:\s*(.+)", raw, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        hey_line = next((l for l in lines if l.startswith("Hey")), None)
        text = hey_line if hey_line else (lines[-1] if lines else raw.strip())
    text = text.strip("\"'").strip()
    return text


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Icebreaker Generator", layout="wide")

st.title("Icebreaker Generator")
st.caption("View your Supabase ice table, generate icebreakers for contacts that don't have one yet.")

# -- Session state defaults --
if "ice_data" not in st.session_state:
    st.session_state["ice_data"] = None
if "stop_requested" not in st.session_state:
    st.session_state["stop_requested"] = False
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = SYSTEM_PROMPT
if "user_prompt_template" not in st.session_state:
    st.session_state["user_prompt_template"] = DEFAULT_USER_TEMPLATE
if "queue" not in st.session_state:
    st.session_state["queue"] = []
if "data_source" not in st.session_state:
    st.session_state["data_source"] = "supabase"

# -- Sidebar: API keys and settings --
with st.sidebar:
    st.header("API Keys")

    supabase_url = st.text_input(
        "Supabase URL",
        value=os.environ.get("SUPABASE_URL", "https://kogjdgxvrxfxqbkchxpi.supabase.co"),
        help="Your Supabase project URL. Override with env SUPABASE_URL.",
    )
    supabase_key = st.text_input(
        "Supabase Anon Key",
        type="password",
        value=os.environ.get("SUPABASE_ANON_KEY", ""),
        help="Your Supabase anonymous/public key. Override with env SUPABASE_ANON_KEY.",
    )
    cerebras_key = st.text_input(
        "Cerebras API Key",
        type="password",
        value=os.environ.get("CEREBRAS_API_KEY", ""),
        help="Your Cerebras API key. Override with env CEREBRAS_API_KEY.",
    )

    st.divider()
    st.header("Settings")
    delay = st.slider("Delay between calls (seconds)", min_value=0, max_value=10, value=2)

    with st.expander("Edit AI prompt"):
        st.session_state["system_prompt"] = st.text_area(
            "System prompt",
            value=st.session_state.get("system_prompt", SYSTEM_PROMPT),
            height=200,
            help="Instructions for the AI. Used for every contact.",
        )
        st.session_state["user_prompt_template"] = st.text_area(
            "User prompt template",
            value=st.session_state.get("user_prompt_template", DEFAULT_USER_TEMPLATE),
            height=120,
            help="Use placeholders: {first_name}, {company}, {title}, {headline}, {city}, {country}, {industry}, {employees}, {website}, {keywords}",
        )

    st.divider()
    st.header("Ice table")
    ice_table_name = st.text_input(
        "Ice table name",
        value=os.environ.get("ICE_TABLE_NAME", "ice"),
        help="Supabase table to load from and save icebreakers to. Override with env ICE_TABLE_NAME.",
    )
    on_conflict_column = st.text_input(
        "Upsert conflict column",
        value="email",
        help="Unique column for merge (e.g. email). Must match a unique constraint on the ice table.",
    )
    st.header("Contacts table")
    contacts_table_name = st.text_input(
        "Contacts table name",
        value=os.environ.get("CONTACTS_TABLE_NAME", "contacts"),
        help="Supabase table to load contacts from (used in From contacts tab). Override with env CONTACTS_TABLE_NAME.",
    )
    max_contacts_to_load = st.number_input(
        "Max contacts to load",
        min_value=1,
        max_value=10000,
        value=2000,
        help="Max rows to fetch from the contacts table when using From contacts.",
    )

keys_ready = bool(supabase_url and supabase_key and cerebras_key)
supabase_ready = bool(supabase_url and supabase_key)

if not supabase_ready:
    st.info("Enter your Supabase URL and Anon Key in the sidebar to get started.")
    st.stop()

st.caption(f"**Ice table:** fill nulls from Supabase table **{ice_table_name}**, or add **new** rows from the **contacts** table **{contacts_table_name}** (emails not already in ice). Configure in sidebar.")

# ===================================================================
# Data source: tabs
# ===================================================================
tab_contacts, tab_supabase, tab_csv, tab_url = st.tabs(["From contacts", "From Supabase", "Upload CSV", "Import from URL"])

# --- Tab: From contacts ---
with tab_contacts:
    if st.button("Load from contacts", type="primary", key="load_from_contacts"):
        with st.spinner("Loading contacts and ice table..."):
            try:
                contacts = load_contacts(supabase_url, supabase_key, contacts_table_name, max_contacts_to_load)
                ice_rows = load_ice_table(supabase_url, supabase_key, ice_table_name)
                ice_by_email = {(r.get("email") or "").strip().lower(): r for r in ice_rows if r.get("email")}
                pending = [
                    c for c in contacts
                    if c.get("email")
                    and not has_valid_icebreaker(ice_by_email.get((c.get("email") or "").strip().lower()))
                ]
                st.session_state["data_source"] = "contacts"
                st.session_state["contacts_pending"] = pending
                st.session_state["stop_requested"] = False
            except Exception as e:
                st.error(f"Failed to load: {e}")
    contacts_pending = st.session_state.get("contacts_pending") if st.session_state.get("data_source") == "contacts" else None
    if contacts_pending is not None:
        st.metric("Contacts not yet in ice table", len(contacts_pending))
        if contacts_pending:
            preview_cols = ["email", "first_name", "company", "title"]
            display_cols = [c for c in preview_cols if any(c in r for r in contacts_pending)]
            if display_cols:
                st.dataframe(pd.DataFrame(contacts_pending)[display_cols].fillna(""), use_container_width=True, hide_index=True, height=250)
        else:
            st.info("All loaded contacts are already in the ice table.")
    else:
        st.info("Click **Load from contacts** to fetch contacts and compare with the ice table.")

# --- Tab: From Supabase ---
with tab_supabase:
    if st.button("Load Data", type="primary", key="load_data"):
        with st.spinner("Loading ice table from Supabase..."):
            try:
                data = load_ice_table(supabase_url, supabase_key, ice_table_name)
                st.session_state["ice_data"] = data
                st.session_state["data_source"] = "supabase"
                st.session_state["stop_requested"] = False
            except Exception as e:
                st.error(f"Failed to load ice table: {e}")

    ice_data = st.session_state.get("ice_data")
    if ice_data is None:
        st.info("Click **Load Data** to fetch the ice table from Supabase.")
    elif not ice_data:
        st.warning("Ice table is empty. Add rows in Supabase, or use **Upload CSV** / **Import from URL** to generate icebreakers.")
    else:
        df = pd.DataFrame(ice_data)
        df["has_icebreaker"] = df["ai_icebreaker"].apply(
            lambda x: bool(x and str(x).strip() and not str(x).strip().startswith(("ERROR", "API_ERROR", "REASONING_EXHAUSTED")))
        )
        total, done = len(df), int(df["has_icebreaker"].sum())
        pending = total - done
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Rows", total)
        col_b.metric("With Icebreaker", done)
        col_c.metric("Pending (null)", pending)
        show_cols = ["email", "first_name", "company", "title", "city", "ai_icebreaker"]
        display_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[display_cols].fillna(""), use_container_width=True, hide_index=True, height=300)
        st.download_button("Download ice table CSV", df.to_csv(index=False), file_name="ice_table_export.csv", mime="text/csv", key="dl_ice")

# --- Tab: Upload CSV ---
with tab_csv:
    uploaded = st.file_uploader("Upload CSV of contacts", type=["csv"], key="csv_upload")
    if uploaded:
        try:
            rows = parse_csv_to_rows(uploaded)
            st.session_state["queue"] = rows
            st.session_state["data_source"] = "csv"
            st.success(f"Loaded **{len(rows)}** contacts (with email). Preview below.")
            if rows:
                st.dataframe(pd.DataFrame(rows[:50]), use_container_width=True, hide_index=True, height=250)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

# --- Tab: Import from URL ---
with tab_url:
    url_input = st.text_input("URL to contacts (JSON or CSV)", placeholder="https://...", key="url_input")
    if st.button("Fetch and load", key="fetch_url"):
        if not url_input.strip():
            st.warning("Enter a URL.")
        else:
            with st.spinner("Fetching..."):
                try:
                    rows = fetch_url_to_rows(url_input.strip())
                    st.session_state["queue"] = rows
                    st.session_state["data_source"] = "url"
                    st.success(f"Loaded **{len(rows)}** contacts (with email). Preview below.")
                    if rows:
                        st.dataframe(pd.DataFrame(rows[:50]), use_container_width=True, hide_index=True, height=250)
                except Exception as e:
                    st.error(f"Failed to fetch or parse: {e}")

# ===================================================================
# Compute to_process: from Supabase (pending) or from queue (CSV/URL)
# ===================================================================
to_process = []
data_source = st.session_state.get("data_source", "supabase")

if data_source == "supabase":
    ice_data = st.session_state.get("ice_data")
    if ice_data:
        to_process = [
            row for row in ice_data
            if not row.get("ai_icebreaker") or not str(row.get("ai_icebreaker", "")).strip()
            or str(row["ai_icebreaker"]).strip().startswith(("ERROR", "API_ERROR", "REASONING_EXHAUSTED"))
        ]
elif data_source == "contacts":
    to_process = st.session_state.get("contacts_pending") or []
else:
    to_process = st.session_state.get("queue") or []

pending_count = len(to_process)

# ===================================================================
# Generate Icebreakers (unified: runs over to_process, saves to ice table)
# ===================================================================
st.divider()

if not keys_ready:
    st.warning("Enter your Cerebras API Key in the sidebar to enable generation.")
elif pending_count == 0:
    if data_source == "supabase":
        st.success("All rows already have icebreakers, or load data first.")
    elif data_source == "contacts":
        st.info("Click **Load from contacts** in the From contacts tab, or all loaded contacts are already in the ice table.")
    else:
        st.info("Load contacts via CSV or URL first, then generate.")
else:
    col_gen, col_stop = st.columns([1, 1])
    with col_gen:
        generate_btn = st.button(
            f"Generate {pending_count} icebreakers and save to ice table",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("is_running", False),
            key="gen_btn",
        )
    with col_stop:
        stop_btn = st.button("Stop", use_container_width=True, type="secondary", key="stop_btn")

    if stop_btn:
        st.session_state["stop_requested"] = True

    if generate_btn:
        st.session_state["stop_requested"] = False
        st.session_state["is_running"] = True

        progress_bar = st.progress(0, text="Starting...")
        log_area = st.container()
        results = []

        for i, row in enumerate(to_process):
            if st.session_state.get("stop_requested", False):
                with log_area:
                    st.warning(f"Stopped by user after {i} contacts.")
                break

            name = row.get("first_name") or "Unknown"
            email = (row.get("email") or "?").strip()
            email_display = email or "?"
            company = row.get("company") or ""
            progress_bar.progress(i / len(to_process), text=f"[{i+1}/{len(to_process)}] {name} ({email_display})")

            # Live check by email: skip if ice table already has a valid icebreaker
            try:
                ice_row = get_ice_row_by_email(supabase_url, supabase_key, ice_table_name, email)
                if has_valid_icebreaker(ice_row):
                    with log_area:
                        st.info(f"Skipped (already has icebreaker): {email_display}")
                    results.append({"name": name, "email": email_display, "company": company, "icebreaker": "[Skipped]", "saved": False})
                    continue
            except Exception:
                pass  # Proceed to generate if check fails (e.g. network)

            try:
                icebreaker = generate_icebreaker(cerebras_key, row)
            except Exception as e:
                icebreaker = f"API_ERROR: {e}"

            try:
                upsert_to_ice(supabase_url, supabase_key, ice_table_name, row, icebreaker, on_conflict_column)
                saved = True
            except Exception as e:
                saved = False
                icebreaker = f"{icebreaker} | SAVE_ERROR: {e}"

            results.append({"name": name, "email": email_display, "company": company, "icebreaker": icebreaker, "saved": saved})

            with log_area:
                if icebreaker.startswith(("ERROR", "API_ERROR", "REASONING_EXHAUSTED")):
                    st.warning(f"**{name}**: {icebreaker}")
                else:
                    st.success(f"**{name}**: {icebreaker} (saved to ice table)")

            if delay > 0 and i < len(to_process) - 1:
                time.sleep(delay)

        progress_bar.progress(1.0, text="Done!")
        st.session_state["is_running"] = False

        if data_source == "supabase":
            try:
                st.session_state["ice_data"] = load_ice_table(supabase_url, supabase_key, ice_table_name)
            except Exception:
                pass

        if results:
            st.divider()
            st.subheader("Run Results")
            st.caption("**Saved** = written to Supabase ice table.")
            rdf = pd.DataFrame(results)
            success_count = int(rdf["saved"].sum())
            error_count = len(rdf) - success_count
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Processed", len(rdf))
            rc2.metric("Saved to ice table", success_count)
            rc3.metric("Errors", error_count)
            st.dataframe(rdf, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download run results as CSV",
                data=rdf.to_csv(index=False),
                file_name="icebreaker_run_results.csv",
                mime="text/csv",
                key="dl_results",
            )
