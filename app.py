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


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
def supabase_headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def load_ice_table(url: str, key: str, limit: int = 2000) -> list[dict]:
    """Load the full ice table from Supabase."""
    resp = requests.get(
        f"{url}/rest/v1/ice",
        params={"select": "*", "limit": str(limit)},
        headers=supabase_headers(key),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def load_contacts(url: str, key: str) -> list[dict]:
    """Load contacts that have an email address."""
    resp = requests.get(
        f"{url}/rest/v1/contacts",
        params={"select": "*", "email": "not.is.null", "limit": "2000"},
        headers=supabase_headers(key),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def sync_contacts_to_ice(url: str, key: str, contacts: list[dict], ice_emails: set) -> int:
    """Copy contacts that aren't in the ice table yet (with null ai_icebreaker)."""
    headers = supabase_headers(key)
    headers["Prefer"] = "resolution=merge-duplicates"
    synced = 0

    for c in contacts:
        email = c.get("email")
        if not email or email in ice_emails:
            continue

        body = {
            "ai_icebreaker": None,
            "first_name": c.get("first_name", ""),
            "last_name": c.get("last_name", ""),
            "full_name": c.get("full_name", ""),
            "email": email,
            "title": c.get("title", ""),
            "headline": c.get("headline", ""),
            "seniority": c.get("seniority", ""),
            "department": c.get("department", ""),
            "city": c.get("city", ""),
            "country": c.get("country", ""),
            "linkedin_url": c.get("linkedin_url", ""),
            "company": c.get("company", ""),
            "website": c.get("company_website") or c.get("website", ""),
            "company_linkedin": c.get("company_linkedin", ""),
            "industry": c.get("industry", ""),
            "employees": c.get("employees_count") or c.get("employees", ""),
            "keywords": c.get("keywords", ""),
        }
        resp = requests.post(f"{url}/rest/v1/ice", json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        synced += 1

    return synced


def upsert_to_ice(url: str, key: str, row: dict, icebreaker: str) -> None:
    """UPSERT a row into the ice table with the generated icebreaker."""
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

    resp = requests.post(f"{url}/rest/v1/ice", json=body, headers=headers, timeout=30)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Cerebras API
# ---------------------------------------------------------------------------
def build_user_prompt(row: dict) -> str:
    """Build the per-contact user prompt from ice row fields."""
    def g(key: str) -> str:
        v = row.get(key)
        return str(v).strip() if v else ""

    return (
        "Write one icebreaker for this person. Pick the single most specific detail.\n\n"
        f"First Name: {g('first_name')}\n"
        f"Title: {g('title')}\n"
        f"Headline: {g('headline')}\n"
        f"Seniority: {g('seniority')}\n"
        f"Department: {g('department')}\n"
        f"City: {g('city')}\n"
        f"Country: {g('country')}\n"
        f"Company: {g('company')}\n"
        f"Industry: {g('industry')}\n"
        f"Employees: {g('employees')}\n"
        f"Website: {g('website')}\n"
        f"Keywords: {g('keywords')}"
    )


def generate_icebreaker(cerebras_key: str, row: dict) -> str:
    """Call Cerebras API and return the icebreaker text."""
    user_prompt = build_user_prompt(row)

    resp = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {cerebras_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-oss-120b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
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

# -- Sidebar: API keys and settings --
with st.sidebar:
    st.header("API Keys")

    supabase_url = st.text_input(
        "Supabase URL",
        value="https://kogjdgxvrxfxqbkchxpi.supabase.co",
        help="Your Supabase project URL",
    )
    supabase_key = st.text_input(
        "Supabase Anon Key",
        type="password",
        help="Your Supabase anonymous/public key",
    )
    cerebras_key = st.text_input(
        "Cerebras API Key",
        type="password",
        help="Your Cerebras API key",
    )

    st.divider()
    st.header("Settings")
    delay = st.slider("Delay between calls (seconds)", min_value=0, max_value=10, value=2)

keys_ready = bool(supabase_url and supabase_key and cerebras_key)
supabase_ready = bool(supabase_url and supabase_key)

if not supabase_ready:
    st.info("Enter your Supabase URL and Anon Key in the sidebar to get started.")
    st.stop()

# ===================================================================
# ROW 1: Load Data + Sync Contacts buttons
# ===================================================================
col_load, col_sync = st.columns([1, 1])

with col_load:
    if st.button("Load Data", type="primary", use_container_width=True):
        with st.spinner("Loading ice table from Supabase..."):
            try:
                data = load_ice_table(supabase_url, supabase_key)
                st.session_state["ice_data"] = data
                st.session_state["stop_requested"] = False
            except Exception as e:
                st.error(f"Failed to load ice table: {e}")

with col_sync:
    if st.button("Sync New Contacts", use_container_width=True, help="Copy contacts not yet in ice table"):
        with st.spinner("Syncing contacts..."):
            try:
                ice_rows = load_ice_table(supabase_url, supabase_key)
                ice_emails = {r["email"] for r in ice_rows if r.get("email")}
                contacts = load_contacts(supabase_url, supabase_key)
                count = sync_contacts_to_ice(supabase_url, supabase_key, contacts, ice_emails)
                if count > 0:
                    st.success(f"Synced {count} new contacts into ice table. Click 'Load Data' to refresh.")
                else:
                    st.info("All contacts are already in the ice table. Nothing to sync.")
                # Refresh data after sync
                st.session_state["ice_data"] = load_ice_table(supabase_url, supabase_key)
            except Exception as e:
                st.error(f"Failed to sync contacts: {e}")

# ===================================================================
# Show ice table data
# ===================================================================
ice_data = st.session_state.get("ice_data")

if ice_data is None:
    st.info("Click **Load Data** to fetch the ice table from Supabase.")
    st.stop()

if not ice_data:
    st.warning("The ice table is empty. Click **Sync New Contacts** to pull contacts in.")
    st.stop()

# Build dataframe
df = pd.DataFrame(ice_data)

# Determine pending vs done
df["has_icebreaker"] = df["ai_icebreaker"].apply(
    lambda x: bool(x and str(x).strip() and not str(x).strip().startswith(("ERROR", "API_ERROR", "REASONING_EXHAUSTED")))
)
total = len(df)
done = int(df["has_icebreaker"].sum())
pending = total - done

# Metrics
st.divider()
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Rows", total)
col_b.metric("With Icebreaker", done)
col_c.metric("Pending (null)", pending)

# Display table
show_cols = ["email", "first_name", "company", "title", "city", "ai_icebreaker"]
display_cols = [c for c in show_cols if c in df.columns]
st.dataframe(
    df[display_cols].fillna(""),
    use_container_width=True,
    hide_index=True,
    height=400,
)

# CSV download of current data
csv_full = df.to_csv(index=False)
st.download_button(
    label="Download full ice table as CSV",
    data=csv_full,
    file_name="ice_table_export.csv",
    mime="text/csv",
)

# ===================================================================
# Generate Icebreakers
# ===================================================================
st.divider()

if not keys_ready:
    st.warning("Enter your Cerebras API Key in the sidebar to enable generation.")
    st.stop()

if pending == 0:
    st.success("All rows already have icebreakers. Nothing to generate.")
    st.stop()

# Buttons
col_gen, col_stop = st.columns([1, 1])

with col_gen:
    generate_btn = st.button(
        f"Generate {pending} Icebreakers",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get("is_running", False),
    )

with col_stop:
    stop_btn = st.button(
        "Stop",
        use_container_width=True,
        type="secondary",
    )

if stop_btn:
    st.session_state["stop_requested"] = True

if generate_btn:
    st.session_state["stop_requested"] = False
    st.session_state["is_running"] = True

    # Get pending rows (ai_icebreaker is null or empty)
    pending_rows = [
        row for row in ice_data
        if not row.get("ai_icebreaker") or not str(row["ai_icebreaker"]).strip()
        or str(row["ai_icebreaker"]).strip().startswith(("ERROR", "API_ERROR", "REASONING_EXHAUSTED"))
    ]

    progress_bar = st.progress(0, text="Starting...")
    log_area = st.container()
    results = []

    for i, row in enumerate(pending_rows):
        # Check stop flag
        if st.session_state.get("stop_requested", False):
            with log_area:
                st.warning(f"Stopped by user after {i} contacts.")
            break

        name = row.get("first_name") or "Unknown"
        email = row.get("email") or "?"
        company = row.get("company") or ""
        progress_bar.progress(i / len(pending_rows), text=f"[{i+1}/{len(pending_rows)}] {name} ({email})")

        # Generate
        try:
            icebreaker = generate_icebreaker(cerebras_key, row)
        except Exception as e:
            icebreaker = f"API_ERROR: {e}"

        # Save to Supabase
        try:
            upsert_to_ice(supabase_url, supabase_key, row, icebreaker)
            saved = True
        except Exception as e:
            saved = False
            icebreaker = f"{icebreaker} | SAVE_ERROR: {e}"

        results.append({"name": name, "email": email, "company": company, "icebreaker": icebreaker, "saved": saved})

        with log_area:
            if icebreaker.startswith(("ERROR", "API_ERROR", "REASONING_EXHAUSTED")):
                st.warning(f"**{name}**: {icebreaker}")
            else:
                st.success(f"**{name}**: {icebreaker}")

        # Delay
        if delay > 0 and i < len(pending_rows) - 1:
            time.sleep(delay)

    progress_bar.progress(1.0, text="Done!")
    st.session_state["is_running"] = False

    # Refresh ice data so the table updates
    try:
        st.session_state["ice_data"] = load_ice_table(supabase_url, supabase_key)
    except Exception:
        pass

    # Show run results
    if results:
        st.divider()
        st.subheader("Run Results")
        rdf = pd.DataFrame(results)
        success_count = int(rdf["saved"].sum())
        error_count = len(rdf) - success_count

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Processed", len(rdf))
        rc2.metric("Saved", success_count)
        rc3.metric("Errors", error_count)

        st.dataframe(rdf, use_container_width=True, hide_index=True)

        csv_results = rdf.to_csv(index=False)
        st.download_button(
            label="Download run results as CSV",
            data=csv_results,
            file_name="icebreaker_run_results.csv",
            mime="text/csv",
        )
